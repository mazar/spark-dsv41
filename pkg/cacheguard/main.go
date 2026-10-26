package main

/*
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif

static long drop_file_page_cache(const char* path) {
    int fd = open(path, O_RDONLY | O_NOATIME);
    if (fd < 0) {
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            return -1;
        }
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return -1;
    }
    int rc = posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
    close(fd);
    if (rc != 0) {
        return -1;
    }
    return (long)st.st_size;
}
*/
import "C"

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

func expandFiles(paths []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, p := range paths {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if strings.HasPrefix(p, "~/") {
			if home, err := os.UserHomeDir(); err == nil {
				p = filepath.Join(home, p[2:])
			}
		}
		p = os.ExpandEnv(p)
		info, err := os.Stat(p)
		if err != nil {
			matches, globErr := filepath.Glob(p)
			if globErr == nil {
				for _, m := range matches {
					if st, err2 := os.Stat(m); err2 == nil && !st.IsDir() && !seen[m] {
						seen[m] = true
						out = append(out, m)
					}
				}
			}
			continue
		}
		if info.IsDir() {
			entries, err := os.ReadDir(p)
			if err != nil {
				continue
			}
			for _, e := range entries {
				if e.IsDir() {
					continue
				}
				name := e.Name()
				if strings.HasSuffix(name, ".safetensors") || strings.HasSuffix(name, ".bin") || strings.HasSuffix(name, ".rkc") || strings.Contains(name, "rank") {
					full := filepath.Join(p, name)
					if !seen[full] {
						seen[full] = true
						out = append(out, full)
					}
				}
			}
		} else if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

func dropAll(targets []string) (int, int64) {
	files := expandFiles(targets)
	count := 0
	var totalBytes int64
	for _, f := range files {
		cPath := C.CString(f)
		sz := int64(C.drop_file_page_cache(cPath))
		C.free(unsafe.Pointer(cPath))
		if sz > 0 {
			count++
			totalBytes += sz
		}
	}
	return count, totalBytes
}

func readMemAvailableGiB() float64 {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				var kb float64
				_, _ = fmt.Sscanf(fields[1], "%f", &kb)
				return kb / (1024.0 * 1024.0)
			}
		}
	}
	return 0
}

func main() {
	interval := flag.Duration("interval", 15*time.Second, "Interval between POSIX_FADV_DONTNEED sweeps on Engram and Rank Cache files")
	once := flag.Bool("once", false, "Run a single POSIX_FADV_DONTNEED sweep and exit")
	pathsFlag := flag.String("paths", "", "Colon- or comma-separated list of files/directories to evict from Linux page cache")
	flag.Parse()

	var targets []string
	if *pathsFlag != "" {
		for _, part := range strings.FieldsFunc(*pathsFlag, func(r rune) bool { return r == ':' || r == ',' }) {
			if part != "" {
				targets = append(targets, part)
			}
		}
	}
	targets = append(targets, flag.Args()...)

	if len(targets) == 0 {
		home, _ := os.UserHomeDir()
		targets = []string{
			filepath.Join(home, ".cache/huggingface/tensorfold-dsv41/custom-hq/rank-cache"),
			filepath.Join(home, ".cache/huggingface/tensorfold-dsv41/45ee5d0f/rank-cache"),
			filepath.Join(home, ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"),
			filepath.Join(home, ".cache/dsv41-tensorfold/rank-cache-custom-hq"),
			filepath.Join(home, ".cache/dsv41-tensorfold/rank-cache-45ee5d0f"),
			filepath.Join(home, ".cache/dsv41-tensorfold/engram-1bae0e10"),
			filepath.Join(home, "models/deepseek-v41-1017-exl3-2.9bpw/engram"),
			filepath.Join(home, "tfv41-cache/tf-ds-rank"),
			filepath.Join(home, "tfv41-cache/tf-ds-spill"),
		}
	}

	cnt, bytesDropped := dropAll(targets)
	log.Printf("[spark-dsv41-cacheguard] Initial POSIX_FADV_DONTNEED sweep across %d files (%.2f GiB covered, MemAvailable=%.2f GiB)",
		cnt, float64(bytesDropped)/(1024*1024*1024), readMemAvailableGiB())

	if *once {
		return
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	for {
		select {
		case <-sigCh:
			log.Printf("[spark-dsv41-cacheguard] Shutting down cleanly")
			return
		case <-ticker.C:
			dropAll(targets)
		}
	}
}
