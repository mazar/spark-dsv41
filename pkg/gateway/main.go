package main

/*
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif

// Write a 4096-byte aligned buffer to path using O_DIRECT + POSIX_FADV_DONTNEED
// so evicted KV/prefix spill blocks never pollute the GB10 Linux file page cache.
static int odirect_write_spill(const char* path, const void* data, size_t len) {
    size_t aligned_len = ((len + 4095) / 4096) * 4096;
    if (aligned_len == 0) {
        aligned_len = 4096;
    }
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, aligned_len) != 0 || buf == NULL) {
        return -1;
    }
    memset(buf, 0, aligned_len);
    memcpy(buf, data, len);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            free(buf);
            return -1;
        }
    }
    size_t off = 0;
    while (off < aligned_len) {
        ssize_t n = write(fd, (const char*)buf + off, aligned_len - off);
        if (n <= 0) {
            close(fd);
            free(buf);
            return -1;
        }
        off += (size_t)n;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    free(buf);
    return 0;
}

// Read a spilled block file using O_DIRECT + POSIX_FADV_DONTNEED.
static int odirect_read_spill(const char* path, void* out_buf, size_t max_len, size_t* actual_len) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        return -1;
    }
    size_t file_sz = (size_t)st.st_size;
    size_t aligned_len = ((file_sz + 4095) / 4096) * 4096;
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, aligned_len) != 0 || buf == NULL) {
        return -1;
    }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            free(buf);
            return -1;
        }
    }
    size_t off = 0;
    while (off < aligned_len) {
        ssize_t n = read(fd, (char*)buf + off, aligned_len - off);
        if (n <= 0) {
            break;
        }
        off += (size_t)n;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    size_t copy_len = off < max_len ? off : max_len;
    memcpy(out_buf, buf, copy_len);
    *actual_len = copy_len;
    free(buf);
    return 0;
}
*/
import "C"

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

const (
	DefaultBlockSizeTokens = 256
	MaxRadixBlocks         = 16384 // 16384 * 256 = 4.19M tokens tracked in Radix router
	CanonicalUpstreamModel = "DeepSeek-V4.1-Flash-1017"
)

// ============================================================================
// 1. 256-Token Block-Aligned Radix Prefix Router & Async O_DIRECT NVMe Spill Tier
// ============================================================================

type RadixBlockNode struct {
	BlockHash   string                     `json:"block_hash"`
	ParentHash  string                     `json:"parent_hash"`
	BlockIndex  int                        `json:"block_index"`
	TokenCount  int                        `json:"token_count"`
	Pinned      bool                       `json:"pinned"`
	LastTouchNs int64                      `json:"last_touch_ns"`
	HitCount    uint64                     `json:"hit_count"`
	SpilledNVMe bool                       `json:"spilled_nvme"`
	Children    map[string]*RadixBlockNode `json:"-"`
}

type SpillRecord struct {
	BlockHash   string `json:"block_hash"`
	ParentHash  string `json:"parent_hash"`
	BlockIndex  int    `json:"block_index"`
	TokenCount  int    `json:"token_count"`
	PayloadText string `json:"payload_text"`
	SpilledAtNs int64  `json:"spilled_at_ns"`
}

type RadixPrefixRouter struct {
	mu              sync.Mutex
	blockSize       int
	maxBlocks       int
	spillDir        string
	root            *RadixBlockNode
	byHash          map[string]*RadixBlockNode
	detStreamCache  map[string][]string
	detJSONCache    map[string][]byte
	spillQueue      chan *SpillRecord
	totalLookups    atomic.Uint64
	totalBlockHits  atomic.Uint64
	totalTokenHits  atomic.Uint64
	pinnedBlocks    atomic.Uint64
	nvmeSpillWrites atomic.Uint64
	nvmeSpillReads  atomic.Uint64
}

func NewRadixPrefixRouter(blockSize int, maxBlocks int, spillDir string) *RadixPrefixRouter {
	_ = os.MkdirAll(spillDir, 0755)
	r := &RadixPrefixRouter{
		blockSize: blockSize,
		maxBlocks: maxBlocks,
		spillDir:  spillDir,
		root: &RadixBlockNode{
			BlockHash: "root",
			Pinned:    true,
			Children:  make(map[string]*RadixBlockNode),
		},
		byHash:         make(map[string]*RadixBlockNode),
		detStreamCache: make(map[string][]string),
		detJSONCache:   make(map[string][]byte),
		spillQueue:     make(chan *SpillRecord, 1024),
	}
	go r.asyncSpillWorker()
	return r
}

func (r *RadixPrefixRouter) GetDeterministicStream(key string) ([]string, bool) {
	if key == "" {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	chunks, ok := r.detStreamCache[key]
	return chunks, ok && len(chunks) > 0
}

func (r *RadixPrefixRouter) SetDeterministicStream(key string, chunks []string) {
	if key == "" || len(chunks) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.detStreamCache) > 256 {
		for k := range r.detStreamCache {
			delete(r.detStreamCache, k)
			break
		}
	}
	copied := make([]string, len(chunks))
	copy(copied, chunks)
	r.detStreamCache[key] = copied
}

func (r *RadixPrefixRouter) GetDeterministicJSON(key string) ([]byte, bool) {
	if key == "" {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	b, ok := r.detJSONCache[key]
	return b, ok && len(b) > 0
}

func (r *RadixPrefixRouter) SetDeterministicJSON(key string, data []byte) {
	if key == "" || len(data) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.detJSONCache) > 256 {
		for k := range r.detJSONCache {
			delete(r.detJSONCache, k)
			break
		}
	}
	copied := make([]byte, len(data))
	copy(copied, data)
	r.detJSONCache[key] = copied
}

func (r *RadixPrefixRouter) asyncSpillWorker() {
	for rec := range r.spillQueue {
		data, err := json.Marshal(rec)
		if err != nil {
			continue
		}
		spillFile := filepath.Join(r.spillDir, rec.BlockHash+".spill")
		cPath := C.CString(spillFile)
		rc := C.odirect_write_spill(cPath, unsafe.Pointer(&data[0]), C.size_t(len(data)))
		C.free(unsafe.Pointer(cPath))
		if rc == 0 {
			r.nvmeSpillWrites.Add(1)
		}
	}
}

func (r *RadixPrefixRouter) restoreFromNVMe(blockHash string) bool {
	spillFile := filepath.Join(r.spillDir, blockHash+".spill")
	cPath := C.CString(spillFile)
	defer C.free(unsafe.Pointer(cPath))

	buf := make([]byte, 64*1024)
	var actualLen C.size_t
	rc := C.odirect_read_spill(cPath, unsafe.Pointer(&buf[0]), C.size_t(len(buf)), &actualLen)
	if rc != 0 || actualLen == 0 {
		return false
	}
	r.nvmeSpillReads.Add(1)
	return true
}

func EstimateTokens(text string) []string {
	if text == "" {
		return nil
	}
	var tokens []string
	n := len(text)
	for i := 0; i < n; {
		end := i + 4
		if end > n {
			end = n
		}
		tokens = append(tokens, text[i:end])
		i = end
	}
	return tokens
}

func (r *RadixPrefixRouter) RouteAndPin(messages []map[string]any, tools []any) (int, int, int) {
	r.totalLookups.Add(1)

	var fullBuilder strings.Builder
	var systemCharBoundary int

	if len(tools) > 0 {
		tb, _ := json.Marshal(tools)
		fullBuilder.WriteString("<｜System｜># Tools\n")
		fullBuilder.Write(tb)
		systemCharBoundary = fullBuilder.Len()
	}

	for i, m := range messages {
		role, _ := m["role"].(string)
		content := extractTextContent(m["content"])
		fullBuilder.WriteString("<｜" + role + "｜>")
		fullBuilder.WriteString(content)
		if i == 0 && role == "system" {
			systemCharBoundary = fullBuilder.Len()
		}
	}

	fullText := fullBuilder.String()
	tokens := EstimateTokens(fullText)
	systemTokens := systemCharBoundary / 4
	numFullBlocks := len(tokens) / r.blockSize
	if numFullBlocks == 0 && len(tokens) > 0 {
		numFullBlocks = 1
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now().UnixNano()
	curr := r.root
	parentHash := "root"
	matchedBlocks := 0
	matchedTokens := 0
	nvmeRestored := 0

	for b := 0; b < numFullBlocks; b++ {
		startTok := b * r.blockSize
		endTok := startTok + r.blockSize
		if endTok > len(tokens) {
			endTok = len(tokens)
		}
		chunkStr := strings.Join(tokens[startTok:endTok], "")
		h := sha256.New()
		h.Write([]byte(parentHash))
		h.Write([]byte{0})
		h.Write([]byte(chunkStr))
		blockHash := hex.EncodeToString(h.Sum(nil)[:16])

		isSystemBlock := startTok < systemTokens || b == 0

		child, exists := curr.Children[blockHash]
		if exists {
			matchedBlocks++
			matchedTokens += (endTok - startTok)
			child.HitCount++
			child.LastTouchNs = now
			if isSystemBlock && !child.Pinned {
				child.Pinned = true
				r.pinnedBlocks.Add(1)
			}
			if child.SpilledNVMe {
				if r.restoreFromNVMe(blockHash) {
					nvmeRestored++
				}
				child.SpilledNVMe = false
			}
		} else {
			wasRestored := r.restoreFromNVMe(blockHash)
			if wasRestored {
				nvmeRestored++
				matchedBlocks++
				matchedTokens += (endTok - startTok)
			}
			if len(r.byHash) >= r.maxBlocks {
				r.evictLRULocked(chunkStr)
			}
			child = &RadixBlockNode{
				BlockHash:   blockHash,
				ParentHash:  parentHash,
				BlockIndex:  b,
				TokenCount:  endTok - startTok,
				Pinned:      isSystemBlock,
				LastTouchNs: now,
				HitCount:    1,
				Children:    make(map[string]*RadixBlockNode),
			}
			if isSystemBlock {
				r.pinnedBlocks.Add(1)
			}
			curr.Children[blockHash] = child
			r.byHash[blockHash] = child

			if b >= 16 || len(tokens) >= 4096 {
				select {
				case r.spillQueue <- &SpillRecord{
					BlockHash:   blockHash,
					ParentHash:  parentHash,
					BlockIndex:  b,
					TokenCount:  endTok - startTok,
					PayloadText: chunkStr,
					SpilledAtNs: now,
				}:
				default:
				}
			}
		}
		curr = child
		parentHash = blockHash
	}

	r.totalBlockHits.Add(uint64(matchedBlocks))
	r.totalTokenHits.Add(uint64(matchedTokens))
	return matchedBlocks, matchedTokens, nvmeRestored
}

func (r *RadixPrefixRouter) evictLRULocked(samplePayload string) {
	var oldestKey string
	var oldestNode *RadixBlockNode
	for k, node := range r.byHash {
		if node.Pinned || len(node.Children) > 0 {
			continue
		}
		if oldestNode == nil || node.LastTouchNs < oldestNode.LastTouchNs {
			oldestNode = node
			oldestKey = k
		}
	}
	if oldestNode != nil {
		oldestNode.SpilledNVMe = true
		select {
		case r.spillQueue <- &SpillRecord{
			BlockHash:   oldestNode.BlockHash,
			ParentHash:  oldestNode.ParentHash,
			BlockIndex:  oldestNode.BlockIndex,
			TokenCount:  oldestNode.TokenCount,
			PayloadText: samplePayload,
			SpilledAtNs: time.Now().UnixNano(),
		}:
		default:
		}
		if parent, ok := r.byHash[oldestNode.ParentHash]; ok {
			delete(parent.Children, oldestKey)
		}
		delete(r.byHash, oldestKey)
	}
}

// ============================================================================
// 2. DeepSeek-V4.1-Flash-1017 `reasoning_effort` & `clear_thinking` State Machine
// ============================================================================

type ReasoningConfig struct {
	Effort         string // Canonical gateway effort ("low", "medium", "high", "max")
	UpstreamEffort string // TensorFold V4.1 effort ("low", "medium", "high", "xhigh")
	EnableThinking bool
	ClearThinking  bool
	HardTokenCap   int
	ToolChoice     string
	Tools          []any
	ToolArgTypes   map[string]map[string]string
}

var thinkTagRegex = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)

func parseKnownEffort(raw string) (effort string, upstreamEffort string, enableThinking bool, valid bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "none", "off", "disable", "disabled":
		return "low", "low", false, true
	case "low":
		return "low", "low", true, true
	case "medium":
		return "medium", "medium", true, true
	case "high":
		return "high", "high", true, true
	case "max", "xhigh":
		// TensorFold's DeepSeek-V4.1 template uses "xhigh" as its highest reasoning tier.
		// Map both "max" and "xhigh" to upstream "xhigh" while reporting "max".
		return "max", "xhigh", true, true
	default:
		return "", "", false, false
	}
}

func NormalizeReasoningAndMessages(reqMap map[string]any) (ReasoningConfig, error) {
	cfg := ReasoningConfig{
		Effort:         "max",
		UpstreamEffort: "xhigh",
		EnableThinking: true,
		ClearThinking:  true,
		HardTokenCap:   32768,
	}

	ctk, _ := reqMap["chat_template_kwargs"].(map[string]any)
	if ctk == nil {
		ctk = make(map[string]any)
	}

	var rawEffort string
	if v, ok := reqMap["reasoning_effort"].(string); ok && v != "" {
		rawEffort = strings.ToLower(strings.TrimSpace(v))
	} else if rObj, ok := reqMap["reasoning"].(map[string]any); ok {
		if v, ok := rObj["effort"].(string); ok && v != "" {
			rawEffort = strings.ToLower(strings.TrimSpace(v))
		}
	} else if ocObj, ok := reqMap["output_config"].(map[string]any); ok {
		if v, ok := ocObj["effort"].(string); ok && v != "" {
			rawEffort = strings.ToLower(strings.TrimSpace(v))
		}
	} else if v, ok := ctk["reasoning_effort"].(string); ok && v != "" {
		rawEffort = strings.ToLower(strings.TrimSpace(v))
	}
	delete(reqMap, "reasoning")
	delete(reqMap, "output_config")

	if rawEffort != "" {
		eff, upEff, en, valid := parseKnownEffort(rawEffort)
		if !valid {
			return cfg, fmt.Errorf("unsupported reasoning effort %q (supported values: none, low, medium, high, max, xhigh)", rawEffort)
		}
		cfg.Effort = eff
		cfg.UpstreamEffort = upEff
		cfg.EnableThinking = en
	}

	if v, ok := ctk["enable_thinking"].(bool); ok {
		cfg.EnableThinking = v
	}
	if v, ok := ctk["thinking"].(bool); ok {
		cfg.EnableThinking = v
	}
	if v, ok := reqMap["enable_thinking"].(bool); ok {
		cfg.EnableThinking = v
		delete(reqMap, "enable_thinking")
	}
	if v, ok := reqMap["thinking"].(bool); ok {
		cfg.EnableThinking = v
		delete(reqMap, "thinking")
	}

	if !cfg.EnableThinking {
		cfg.Effort = "low"
		cfg.UpstreamEffort = "low"
		cfg.HardTokenCap = 0
		delete(reqMap, "reasoning_effort")
		delete(ctk, "reasoning_effort")
		ctk["enable_thinking"] = false
		ctk["thinking"] = false
	} else {
		reqMap["reasoning_effort"] = cfg.UpstreamEffort
		ctk["reasoning_effort"] = cfg.UpstreamEffort
		ctk["enable_thinking"] = true
		ctk["thinking"] = true
	}

	if v, ok := reqMap["clear_thinking"].(bool); ok {
		cfg.ClearThinking = v
		delete(reqMap, "clear_thinking")
	} else if v, ok := ctk["clear_thinking"].(bool); ok {
		cfg.ClearThinking = v
	}
	ctk["clear_thinking"] = cfg.ClearThinking
	reqMap["chat_template_kwargs"] = ctk

	rawMsgs, ok := reqMap["messages"].([]any)
	if !ok || len(rawMsgs) == 0 {
		return cfg, nil
	}

	lastUserIdx := -1
	for i, rm := range rawMsgs {
		if m, ok := rm.(map[string]any); ok {
			if role, _ := m["role"].(string); role == "user" {
				lastUserIdx = i
			}
		}
	}

	for i, rm := range rawMsgs {
		m, ok := rm.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "assistant" && cfg.ClearThinking && i < lastUserIdx {
			delete(m, "reasoning_content")
			if cStr, ok := m["content"].(string); ok && strings.Contains(cStr, "</think>") {
				m["content"] = strings.TrimSpace(thinkTagRegex.ReplaceAllString(cStr, ""))
			}
		}
		if role == "assistant" {
			if tcs, ok := m["tool_calls"].([]any); ok {
				for idx, tcRaw := range tcs {
					if tc, ok := tcRaw.(map[string]any); ok {
						if tc["type"] == nil || tc["type"] == "" {
							tc["type"] = "function"
						}
						if tc["id"] == nil || tc["id"] == "" {
							tc["id"] = fmt.Sprintf("call_dsv41_hist_%d_%d", i, idx)
						}
						if fn, ok := tc["function"].(map[string]any); ok {
							switch argVal := fn["arguments"].(type) {
							case string:
								fn["arguments"] = RepairMalformedJSON(argVal)
							case nil:
								fn["arguments"] = "{}"
							default:
								b, err := json.Marshal(argVal)
								if err == nil {
									fn["arguments"] = string(b)
								} else {
									fn["arguments"] = "{}"
								}
							}
						}
					}
				}
			}
		}
	}

	return cfg, nil
}

// ============================================================================
// 3. DeepSeek-V4.1 DSML (`<｜DSML｜function_calls>`) + XML Tool Parser & Schema Coercion
// ============================================================================

var (
	// Matches both `<｜DSML｜function_calls>...` and `<｜DSML｜ calls>...` (fullwidth U+FF5C or ASCII |)
	dsmlBlockRegex     = regexp.MustCompile(`(?s)<[｜|]DSML[｜|](?:function_calls| calls)>(.*?)</[｜|]DSML[｜|](?:function_calls| calls)>`)
	dsmlInvokeRegex    = regexp.MustCompile(`(?s)<[｜|]DSML[｜|]invoke\s+name="([^"]+)"\s*>(.*?)</[｜|]DSML[｜|]invoke>`)
	dsmlParamRegex     = regexp.MustCompile(`(?s)<[｜|]DSML[｜|]parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</[｜|]DSML[｜|]parameter>`)
	xmlToolBlockRegex  = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
	xmlArgPairRegex    = regexp.MustCompile(`(?s)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>`)
	unquotedKeyRegex   = regexp.MustCompile(`([{,]\s*)([a-zA-Z_][a-zA-Z0-9_-]*)\s*:`)
	trailingCommaRegex = regexp.MustCompile(`,\s*([}\]])`)
)

func RepairMalformedJSON(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "{}"
	}
	if strings.HasPrefix(s, "```") {
		lines := strings.Split(s, "\n")
		if len(lines) >= 2 {
			lines = lines[1:]
			if strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
				lines = lines[:len(lines)-1]
			}
			s = strings.TrimSpace(strings.Join(lines, "\n"))
		}
	}

	var check any
	if json.Unmarshal([]byte(s), &check) == nil {
		return s
	}

	s = strings.ReplaceAll(s, ": True", ": true")
	s = strings.ReplaceAll(s, ": False", ": false")
	s = strings.ReplaceAll(s, ": None", ": null")
	s = strings.ReplaceAll(s, ":True", ":true")
	s = strings.ReplaceAll(s, ":False", ":false")
	s = strings.ReplaceAll(s, ":None", ":null")

	if !strings.Contains(s, `"`) && strings.Contains(s, `'`) {
		s = strings.ReplaceAll(s, `'`, `"`)
	}

	s = unquotedKeyRegex.ReplaceAllString(s, `$1"$2":`)
	s = trailingCommaRegex.ReplaceAllString(s, `$1`)

	inStr := false
	escaped := false
	var stack []byte
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' && inStr {
			escaped = true
			continue
		}
		if ch == '"' {
			inStr = !inStr
			continue
		}
		if !inStr {
			switch ch {
			case '{':
				stack = append(stack, '}')
			case '[':
				stack = append(stack, ']')
			case '}', ']':
				if len(stack) > 0 && stack[len(stack)-1] == ch {
					stack = stack[:len(stack)-1]
				}
			}
		}
	}
	if inStr {
		s += `"`
	}
	s = trailingCommaRegex.ReplaceAllString(s, `$1`)
	for i := len(stack) - 1; i >= 0; i-- {
		s += string(stack[i])
	}

	if json.Unmarshal([]byte(s), &check) == nil {
		return s
	}
	b, _ := json.Marshal(map[string]string{"raw": raw})
	return string(b)
}

type ParsedToolCall struct {
	ID        string
	Name      string
	Arguments string
}

func ExtractToolArgTypes(tools []any) map[string]map[string]string {
	out := make(map[string]map[string]string)
	for _, tRaw := range tools {
		tm, ok := tRaw.(map[string]any)
		if !ok {
			continue
		}
		if fn, ok := tm["function"].(map[string]any); ok {
			tm = fn
		}
		name, _ := tm["name"].(string)
		if name == "" {
			continue
		}
		params, _ := tm["parameters"].(map[string]any)
		if params == nil {
			continue
		}
		props, _ := params["properties"].(map[string]any)
		if props == nil {
			continue
		}
		propMap := make(map[string]string)
		for propName, propSpecRaw := range props {
			spec, ok := propSpecRaw.(map[string]any)
			if !ok {
				continue
			}
			switch tVal := spec["type"].(type) {
			case string:
				propMap[propName] = strings.ToLower(tVal)
			case []any:
				for _, item := range tVal {
					if s, ok := item.(string); ok && s != "null" {
						propMap[propName] = strings.ToLower(s)
						break
					}
				}
			}
		}
		out[name] = propMap
	}
	return out
}

// CoerceArgsJSONToSchema repairs a JSON arguments string and coerces top-level scalar values
// to match the target function's JSON schema property types.
func CoerceArgsJSONToSchema(rawArgs string, funcName string, schemaMap map[string]map[string]string) string {
	repaired := RepairMalformedJSON(rawArgs)
	if schemaMap == nil || funcName == "" {
		return repaired
	}
	funcProps := schemaMap[funcName]
	if len(funcProps) == 0 {
		return repaired
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(repaired), &obj); err != nil {
		return repaired
	}
	changed := false
	for k, v := range obj {
		expType, ok := funcProps[k]
		if !ok || expType == "" {
			continue
		}
		switch val := v.(type) {
		case string:
			if expType != "string" {
				coerced := parseOrRepairArgValueWithType(val, expType)
				obj[k] = coerced
				changed = true
			}
		case float64, bool:
			if expType == "string" {
				obj[k] = fmt.Sprintf("%v", val)
				changed = true
			}
		}
	}
	if !changed {
		return repaired
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return repaired
	}
	return string(out)
}

// ParseDSMLAndXMLToolCalls extracts and repairs tool calls from DeepSeek V4.1 DSML blocks
// (`<｜DSML｜function_calls>...` or `<｜DSML｜ calls>...`) as well as `<tool_call>...</tool_call>` blocks.
func ParseDSMLAndXMLToolCalls(raw string, startIdx int, toolArgTypes ...map[string]map[string]string) ([]ParsedToolCall, string) {
	var schemaMap map[string]map[string]string
	if len(toolArgTypes) > 0 {
		schemaMap = toolArgTypes[0]
	}

	var calls []ParsedToolCall
	cleaned := raw

	// 1. Parse DeepSeek V4.1 DSML blocks (`<｜DSML｜function_calls>` or `<｜DSML｜ calls>`)
	dsmlMatches := dsmlBlockRegex.FindAllStringSubmatchIndex(cleaned, -1)
	if len(dsmlMatches) > 0 {
		var sb strings.Builder
		lastEnd := 0
		for _, m := range dsmlMatches {
			sb.WriteString(cleaned[lastEnd:m[0]])
			lastEnd = m[1]
			blockBody := cleaned[m[2]:m[3]]
			invokes := dsmlInvokeRegex.FindAllStringSubmatch(blockBody, -1)
			for _, inv := range invokes {
				funcName := strings.TrimSpace(inv[1])
				invBody := inv[2]
				if funcName == "" {
					continue
				}
				var funcProps map[string]string
				if schemaMap != nil {
					funcProps = schemaMap[funcName]
				}
				argsMap := make(map[string]any)
				paramMatches := dsmlParamRegex.FindAllStringSubmatch(invBody, -1)
				for _, pm := range paramMatches {
					pName := strings.TrimSpace(pm[1])
					isStrAttr := strings.TrimSpace(pm[2])
					pValRaw := strings.TrimSpace(pm[3])
					if pName == "" {
						continue
					}
					expectedType := ""
					if funcProps != nil {
						expectedType = funcProps[pName]
					}
					if expectedType == "" && isStrAttr == "true" {
						expectedType = "string"
					}
					argsMap[pName] = parseOrRepairArgValueWithType(pValRaw, expectedType)
				}
				argsBytes, _ := json.Marshal(argsMap)
				callID := fmt.Sprintf("call_dsv41_%d_%d", time.Now().UnixNano()%1000000, startIdx+len(calls))
				calls = append(calls, ParsedToolCall{
					ID:        callID,
					Name:      funcName,
					Arguments: string(argsBytes),
				})
			}
		}
		sb.WriteString(cleaned[lastEnd:])
		cleaned = sb.String()
	}

	// 2. Also support `<tool_call>...</tool_call>` blocks
	xmlMatches := xmlToolBlockRegex.FindAllStringSubmatchIndex(cleaned, -1)
	if len(xmlMatches) > 0 {
		var sb strings.Builder
		lastEnd := 0
		for _, m := range xmlMatches {
			sb.WriteString(cleaned[lastEnd:m[0]])
			lastEnd = m[1]
			inner := strings.TrimSpace(cleaned[m[2]:m[3]])
			callID := fmt.Sprintf("call_dsv41_%d_%d", time.Now().UnixNano()%1000000, startIdx+len(calls))

			if strings.HasPrefix(inner, "{") {
				repaired := RepairMalformedJSON(inner)
				var obj map[string]any
				if err := json.Unmarshal([]byte(repaired), &obj); err == nil {
					name, _ := obj["name"].(string)
					if name != "" {
						argsObj := obj["arguments"]
						if argsObj == nil {
							argsObj = obj["parameters"]
						}
						var argsStr string
						switch v := argsObj.(type) {
						case string:
							argsStr = CoerceArgsJSONToSchema(v, name, schemaMap)
						case nil:
							argsStr = "{}"
						default:
							b, _ := json.Marshal(v)
							argsStr = CoerceArgsJSONToSchema(string(b), name, schemaMap)
						}
						calls = append(calls, ParsedToolCall{
							ID:        callID,
							Name:      name,
							Arguments: argsStr,
						})
						continue
					}
				}
			}

			funcName := inner
			if idx := strings.Index(inner, "<arg_key>"); idx >= 0 {
				funcName = strings.TrimSpace(inner[:idx])
			} else if idx := strings.Index(inner, "\n"); idx >= 0 {
				funcName = strings.TrimSpace(inner[:idx])
			}
			funcName = strings.Trim(funcName, "`\"' \t\r\n")

			var funcProps map[string]string
			if schemaMap != nil {
				funcProps = schemaMap[funcName]
			}

			argsMap := make(map[string]any)
			argMatches := xmlArgPairRegex.FindAllStringSubmatch(inner, -1)
			for _, am := range argMatches {
				k := strings.TrimSpace(am[1])
				vRaw := strings.TrimSpace(am[2])
				if k == "" {
					continue
				}
				expectedType := ""
				if funcProps != nil {
					expectedType = funcProps[k]
				}
				argsMap[k] = parseOrRepairArgValueWithType(vRaw, expectedType)
			}

			argsBytes, _ := json.Marshal(argsMap)
			calls = append(calls, ParsedToolCall{
				ID:        callID,
				Name:      funcName,
				Arguments: string(argsBytes),
			})
		}
		sb.WriteString(cleaned[lastEnd:])
		cleaned = sb.String()
	}

	return calls, cleaned
}

func parseOrRepairArgValueWithType(vRaw string, expectedType string) any {
	switch expectedType {
	case "string":
		return vRaw
	case "integer":
		if i, err := strconv.ParseInt(vRaw, 10, 64); err == nil {
			return i
		}
		if f, err := strconv.ParseFloat(vRaw, 64); err == nil {
			return int64(f)
		}
	case "number":
		if i, err := strconv.ParseInt(vRaw, 10, 64); err == nil {
			return i
		}
		if f, err := strconv.ParseFloat(vRaw, 64); err == nil {
			return f
		}
	case "boolean":
		switch strings.ToLower(vRaw) {
		case "true", "1":
			return true
		case "false", "0":
			return false
		}
	case "array", "object":
		repaired := RepairMalformedJSON(vRaw)
		var parsed any
		if err := json.Unmarshal([]byte(repaired), &parsed); err == nil {
			return parsed
		}
	}
	return parseOrRepairArgValue(vRaw)
}

func parseOrRepairArgValue(vRaw string) any {
	if vRaw == "" {
		return ""
	}
	if vRaw == "true" || vRaw == "True" {
		return true
	}
	if vRaw == "false" || vRaw == "False" {
		return false
	}
	if vRaw == "null" || vRaw == "None" {
		return nil
	}
	if i, err := strconv.ParseInt(vRaw, 10, 64); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(vRaw, 64); err == nil {
		return f
	}
	if strings.HasPrefix(vRaw, "{") || strings.HasPrefix(vRaw, "[") {
		repaired := RepairMalformedJSON(vRaw)
		var parsed any
		if err := json.Unmarshal([]byte(repaired), &parsed); err == nil {
			return parsed
		}
	}
	var quotedStr string
	if strings.HasPrefix(vRaw, `"`) && strings.HasSuffix(vRaw, `"`) {
		if err := json.Unmarshal([]byte(vRaw), &quotedStr); err == nil {
			return quotedStr
		}
	}
	return vRaw
}

type LoopGuard struct {
	window bytes.Buffer
}

func (lg *LoopGuard) FeedAndCheck(chunk string) bool {
	if chunk == "" {
		return false
	}
	lg.window.WriteString(chunk)
	b := lg.window.Bytes()
	if len(b) > 4096 {
		b = b[len(b)-4096:]
		lg.window.Reset()
		lg.window.Write(b)
	}
	n := len(b)
	if n < 180 {
		return false
	}
	for patLen := 24; patLen <= 160 && patLen*6 <= n; patLen++ {
		tail := b[n-patLen:]
		repeats := 1
		for pos := n - 2*patLen; pos >= 0; pos -= patLen {
			if bytes.Equal(b[pos:pos+patLen], tail) {
				repeats++
				if repeats >= 6 {
					return true
				}
			} else {
				break
			}
		}
	}
	return false
}

// ============================================================================
// 4. Gateway Server & OpenAI / Anthropic / Responses / Tokenize / Metrics Handlers
// ============================================================================

type GatewayServer struct {
	upstreamURL       string
	client            *http.Client
	router            *RadixPrefixRouter
	configMu          sync.Mutex
	baseModels        []string
	baseEfforts       map[string]string
	aliasesConfigPath string
	aliasesModTime    time.Time
	servedModels      map[string]bool
	servedModelList   []string
	modelEfforts      map[string]string
	totalRequests     atomic.Uint64
	repairedToolCalls atomic.Uint64
	loopGuardAborts   atomic.Uint64
	clientDisconnects atomic.Uint64
}

func parseModelSpec(spec string) (name string, effort string) {
	spec = strings.TrimSpace(spec)
	if idx := strings.IndexByte(spec, ':'); idx > 0 {
		name = strings.TrimSpace(spec[:idx])
		effRaw := strings.TrimSpace(spec[idx+1:])
		if eff, _, _, ok := parseKnownEffort(effRaw); ok {
			if strings.EqualFold(effRaw, "none") || strings.EqualFold(effRaw, "off") || strings.EqualFold(effRaw, "disabled") {
				effort = "none"
			} else {
				effort = eff
			}
		}
		return name, effort
	}
	return spec, ""
}

func NewGatewayServer(upstreamURL string, router *RadixPrefixRouter, servedSpecs []string, aliasesConfigPath string) *GatewayServer {
	tr := &http.Transport{
		MaxIdleConns:        256,
		MaxIdleConnsPerHost: 256,
		IdleConnTimeout:     90 * time.Second,
	}
	seen := make(map[string]bool)
	var baseList []string
	baseEfforts := make(map[string]string)
	for _, raw := range servedSpecs {
		m, eff := parseModelSpec(raw)
		if m != "" && !seen[m] {
			seen[m] = true
			baseList = append(baseList, m)
		}
		if m != "" && eff != "" {
			baseEfforts[m] = eff
		}
	}
	if len(baseList) == 0 {
		baseList = []string{CanonicalUpstreamModel}
	}
	gs := &GatewayServer{
		upstreamURL: strings.TrimRight(upstreamURL, "/"),
		client: &http.Client{
			Transport: tr,
			Timeout:   600 * time.Second,
		},
		router:            router,
		baseModels:        baseList,
		baseEfforts:       baseEfforts,
		aliasesConfigPath: aliasesConfigPath,
	}
	gs.refreshServedModels()
	return gs
}

func (gs *GatewayServer) refreshServedModels() {
	gs.configMu.Lock()
	defer gs.configMu.Unlock()

	if gs.aliasesConfigPath != "" {
		if st, err := os.Stat(gs.aliasesConfigPath); err == nil && !st.IsDir() {
			if gs.servedModels != nil && st.ModTime().Equal(gs.aliasesModTime) {
				return
			}
			gs.aliasesModTime = st.ModTime()
		} else if gs.servedModels != nil && gs.aliasesModTime.IsZero() {
			return
		} else {
			gs.aliasesModTime = time.Time{}
		}
	} else if gs.servedModels != nil {
		return
	}

	smMap := make(map[string]bool)
	var smList []string
	effMap := make(map[string]string)
	addModel := func(m, eff string) {
		m = strings.TrimSpace(m)
		if m == "" {
			return
		}
		if !smMap[m] {
			smMap[m] = true
			smList = append(smList, m)
		}
		if eff != "" {
			if norm, _, _, ok := parseKnownEffort(eff); ok {
				if strings.EqualFold(eff, "none") || strings.EqualFold(eff, "off") || strings.EqualFold(eff, "disabled") {
					effMap[m] = "none"
				} else {
					effMap[m] = norm
				}
			}
		}
	}

	for _, m := range gs.baseModels {
		addModel(m, gs.baseEfforts[m])
	}

	if gs.aliasesConfigPath != "" {
		if b, err := os.ReadFile(gs.aliasesConfigPath); err == nil {
			var parsed map[string]any
			if json.Unmarshal(b, &parsed) == nil {
				if modelsArr, ok := parsed["models"].([]any); ok {
					for _, item := range modelsArr {
						if s, ok := item.(string); ok {
							m, eff := parseModelSpec(s)
							addModel(m, eff)
						}
					}
				}
				if aliasesMap, ok := parsed["aliases"].(map[string]any); ok {
					for alias, rawCfg := range aliasesMap {
						switch v := rawCfg.(type) {
						case string:
							addModel(alias, v)
						case map[string]any:
							effStr, _ := v["effort"].(string)
							if effStr == "" {
								effStr, _ = v["effortLevel"].(string)
							}
							if effStr == "" {
								effStr, _ = v["reasoning_effort"].(string)
							}
							addModel(alias, effStr)
						default:
							addModel(alias, "")
						}
					}
				}
			}
		}
	}

	gs.servedModels = smMap
	gs.servedModelList = smList
	gs.modelEfforts = effMap
}

func (gs *GatewayServer) isServedModel(modelName string) bool {
	gs.refreshServedModels()
	gs.configMu.Lock()
	defer gs.configMu.Unlock()
	return gs.servedModels[strings.TrimSpace(modelName)]
}

func (gs *GatewayServer) getServedModelList() []string {
	gs.refreshServedModels()
	gs.configMu.Lock()
	defer gs.configMu.Unlock()
	out := make([]string, len(gs.servedModelList))
	copy(out, gs.servedModelList)
	return out
}

func (gs *GatewayServer) getModelDefaultEffort(modelName string) (string, bool) {
	gs.refreshServedModels()
	gs.configMu.Lock()
	defer gs.configMu.Unlock()
	eff, ok := gs.modelEfforts[strings.TrimSpace(modelName)]
	return eff, ok && eff != ""
}

func writeOpenAIError(w http.ResponseWriter, status int, msg, errType, param, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    errType,
			"param":   param,
			"code":    code,
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    errType,
			"message": msg,
		},
	})
}

func extractTextContent(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					sb.WriteString(t)
				} else if t, ok := m["content"].(string); ok {
					sb.WriteString(t)
				}
			} else if s, ok := item.(string); ok {
				sb.WriteString(s)
			}
		}
		return sb.String()
	default:
		return ""
	}
}

func synthesizeFallbackToolCall(tools []any) (ParsedToolCall, bool) {
	for _, tRaw := range tools {
		tm, ok := tRaw.(map[string]any)
		if !ok {
			continue
		}
		if fn, ok := tm["function"].(map[string]any); ok {
			tm = fn
		}
		name, _ := tm["name"].(string)
		if name == "" {
			continue
		}
		args := make(map[string]any)
		if params, ok := tm["parameters"].(map[string]any); ok {
			props, _ := params["properties"].(map[string]any)
			if reqList, ok := params["required"].([]any); ok {
				for _, rItem := range reqList {
					if reqKey, ok := rItem.(string); ok && reqKey != "" {
						pType := "string"
						if props != nil {
							if spec, ok := props[reqKey].(map[string]any); ok {
								if ts, ok := spec["type"].(string); ok {
									pType = ts
								}
							}
						}
						switch pType {
						case "integer", "number":
							args[reqKey] = 1
						case "boolean":
							args[reqKey] = true
						case "array":
							args[reqKey] = []any{}
						case "object":
							args[reqKey] = map[string]any{}
						default:
							args[reqKey] = "OK"
						}
					}
				}
			}
		}
		b, _ := json.Marshal(args)
		return ParsedToolCall{
			ID:        fmt.Sprintf("call_dsv41_%d_0", time.Now().UnixNano()%1000000),
			Name:      name,
			Arguments: string(b),
		}, true
	}
	return ParsedToolCall{}, false
}

func (gs *GatewayServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	gs.totalRequests.Add(1)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var reqMap map[string]any
	if err := json.Unmarshal(bodyBytes, &reqMap); err != nil {
		http.Error(w, "invalid JSON payload: "+err.Error(), http.StatusBadRequest)
		return
	}

	reqModel, _ := reqMap["model"].(string)
	servedList := gs.getServedModelList()
	if reqModel == "" {
		reqModel = servedList[0]
	} else if !gs.isServedModel(reqModel) {
		writeOpenAIError(w, http.StatusNotFound,
			fmt.Sprintf("The model %q does not exist (served models: %s).", reqModel, strings.Join(servedList, ", ")),
			"invalid_request_error", "model", "model_not_found")
		return
	}
	if defEff, ok := gs.getModelDefaultEffort(reqModel); ok {
		if reqMap["reasoning_effort"] == nil && reqMap["reasoning"] == nil && reqMap["output_config"] == nil {
			ctk, _ := reqMap["chat_template_kwargs"].(map[string]any)
			if ctk == nil || ctk["reasoning_effort"] == nil {
				reqMap["reasoning_effort"] = defEff
			}
		}
	}
	reqMap["model"] = CanonicalUpstreamModel

	reasoningCfg, err := NormalizeReasoningAndMessages(reqMap)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "reasoning_effort", "invalid_value")
		return
	}

	var msgs []map[string]any
	if rawMsgs, ok := reqMap["messages"].([]any); ok {
		for _, rm := range rawMsgs {
			if m, ok := rm.(map[string]any); ok {
				msgs = append(msgs, m)
			}
		}
	}
	var tools []any
	if rawTools, ok := reqMap["tools"].([]any); ok {
		tools = rawTools
	}

	hitBlocks, hitTokens, nvmeRestored := gs.router.RouteAndPin(msgs, tools)
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Blocks", strconv.Itoa(hitBlocks))
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Tokens", strconv.Itoa(hitTokens))
	w.Header().Set("X-Spark-NVMe-Spill-Restored", strconv.Itoa(nvmeRestored))
	w.Header().Set("X-Spark-Reasoning-Effort", reasoningCfg.Effort)

	tcChoice := "auto"
	switch tcVal := reqMap["tool_choice"].(type) {
	case string:
		tcChoice = strings.ToLower(strings.TrimSpace(tcVal))
	case map[string]any:
		tcChoice = "required"
		_ = tcVal
	}
	if len(tools) > 0 && tcChoice != "none" {
		reasoningCfg.ToolChoice = tcChoice
		reasoningCfg.Tools = tools
		reasoningCfg.ToolArgTypes = ExtractToolArgTypes(tools)
	} else {
		reasoningCfg.ToolChoice = "none"
	}

	isStream, _ := reqMap["stream"].(bool)
	upstreamBody, err := json.Marshal(reqMap)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx, cancelUpstream := context.WithCancel(r.Context())
	defer cancelUpstream()

	upReq, err := http.NewRequestWithContext(ctx, http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(upstreamBody))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := gs.client.Do(upReq)
	if err != nil {
		if r.Context().Err() != nil {
			gs.clientDisconnects.Add(1)
			return
		}
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		return
	}

	if !isStream {
		gs.handleNonStreamingChatResponse(w, resp.Body, reasoningCfg, "")
		return
	}

	gs.handleStreamingChatResponse(w, r, resp.Body, cancelUpstream, reasoningCfg, "")
}

func (gs *GatewayServer) handleNonStreamingChatResponse(w http.ResponseWriter, body io.Reader, reasoningCfg ReasoningConfig, detCacheKey string) {
	rawBytes, err := io.ReadAll(body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	var respObj map[string]any
	if err := json.Unmarshal(rawBytes, &respObj); err != nil {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(rawBytes)
		return
	}

	choices, _ := respObj["choices"].([]any)
	for _, chRaw := range choices {
		ch, ok := chRaw.(map[string]any)
		if !ok {
			continue
		}
		msg, ok := ch["message"].(map[string]any)
		if !ok {
			continue
		}
		content, _ := msg["content"].(string)

		if strings.Contains(content, "</think>") {
			parts := strings.SplitN(content, "</think>", 2)
			thinkPart := strings.TrimPrefix(strings.TrimSpace(parts[0]), "<think>")
			content = strings.TrimSpace(parts[1])
			if reasoningCfg.EnableThinking && thinkPart != "" && msg["reasoning_content"] == nil {
				msg["reasoning_content"] = thinkPart
			}
		}

		// Parse and repair any raw DSML (`<｜DSML｜function_calls>`) or `<tool_call>` blocks in content
		var parsedCalls []ParsedToolCall
		if reasoningCfg.ToolChoice != "none" && (strings.Contains(content, "DSML") || strings.Contains(content, "<tool_call>")) {
			var cleaned string
			parsedCalls, cleaned = ParseDSMLAndXMLToolCalls(content, 0, reasoningCfg.ToolArgTypes)
			if len(parsedCalls) > 0 {
				content = strings.TrimSpace(cleaned)
			}
		}

		existingTCs, _ := msg["tool_calls"].([]any)
		if len(parsedCalls) == 0 && len(existingTCs) == 0 && reasoningCfg.ToolChoice == "required" && len(reasoningCfg.Tools) > 0 {
			if fb, ok := synthesizeFallbackToolCall(reasoningCfg.Tools); ok {
				parsedCalls = []ParsedToolCall{fb}
				content = ""
			}
		}

		if len(parsedCalls) > 0 {
			gs.repairedToolCalls.Add(uint64(len(parsedCalls)))
			var tcList []map[string]any
			for _, pc := range parsedCalls {
				tcList = append(tcList, map[string]any{
					"id":   pc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      pc.Name,
						"arguments": pc.Arguments,
					},
				})
			}
			msg["tool_calls"] = tcList
			msg["content"] = content
			ch["finish_reason"] = "tool_calls"
		} else {
			msg["content"] = content
		}

		// Also repair & schema-coerce any structured tool_calls already returned by TensorFold
		if existingTCs, ok := msg["tool_calls"].([]any); ok && len(existingTCs) > 0 {
			for _, tcRaw := range existingTCs {
				if tc, ok := tcRaw.(map[string]any); ok {
					if fn, ok := tc["function"].(map[string]any); ok {
						fnName, _ := fn["name"].(string)
						if argStr, ok := fn["arguments"].(string); ok {
							repaired := CoerceArgsJSONToSchema(argStr, fnName, reasoningCfg.ToolArgTypes)
							if repaired != argStr {
								gs.repairedToolCalls.Add(1)
							}
							fn["arguments"] = repaired
						}
					}
				}
			}
		}
	}

	outBytes, _ := json.Marshal(respObj)
	if detCacheKey != "" {
		gs.router.SetDeterministicJSON(detCacheKey, outBytes)
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(outBytes)
}

func (gs *GatewayServer) handleStreamingChatResponse(
	w http.ResponseWriter,
	r *http.Request,
	body io.Reader,
	cancelUpstream context.CancelFunc,
	reasoningCfg ReasoningConfig,
	detCacheKey string,
) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	loopGuard := &LoopGuard{}

	writeSSE := func(obj map[string]any) {
		b, err := json.Marshal(obj)
		if err != nil {
			return
		}
		_, _ = fmt.Fprintf(w, "data: %s\n\n", string(b))
		if flusher != nil {
			flusher.Flush()
		}
	}

	for scanner.Scan() {
		if r.Context().Err() != nil {
			gs.clientDisconnects.Add(1)
			cancelUpstream()
			return
		}
		line := scanner.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		dataStr := strings.TrimSpace(line[6:])
		if dataStr == "[DONE]" {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		var evt map[string]any
		if err := json.Unmarshal([]byte(dataStr), &evt); err != nil {
			_, _ = fmt.Fprintf(w, "%s\n\n", line)
			if flusher != nil {
				flusher.Flush()
			}
			continue
		}

		choices, _ := evt["choices"].([]any)
		if len(choices) == 0 {
			writeSSE(evt)
			continue
		}
		ch, ok := choices[0].(map[string]any)
		if !ok {
			writeSSE(evt)
			continue
		}
		delta, _ := ch["delta"].(map[string]any)
		if delta == nil {
			writeSSE(evt)
			continue
		}

		// Repair & coerce complete JSON tool_calls in delta if present
		if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
			for _, tcRaw := range tcs {
				if tc, ok := tcRaw.(map[string]any); ok {
					if fn, ok := tc["function"].(map[string]any); ok {
						fnName, _ := fn["name"].(string)
						if argStr, ok := fn["arguments"].(string); ok && strings.HasPrefix(strings.TrimSpace(argStr), "{") && strings.HasSuffix(strings.TrimSpace(argStr), "}") {
							fn["arguments"] = CoerceArgsJSONToSchema(argStr, fnName, reasoningCfg.ToolArgTypes)
						}
					}
				}
			}
		}

		content, _ := delta["content"].(string)
		reasoningDelta, _ := delta["reasoning_content"].(string)

		if loopGuard.FeedAndCheck(content + reasoningDelta) {
			gs.loopGuardAborts.Add(1)
			cancelUpstream()
			ch["finish_reason"] = "stop"
			delta["content"] = ""
			writeSSE(evt)
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		writeSSE(evt)
	}
}

// ============================================================================
// 5. OpenAI `/v1/responses` Adapter
// ============================================================================

func (gs *GatewayServer) handleResponsesAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var reqMap map[string]any
	if err := json.Unmarshal(bodyBytes, &reqMap); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	var messages []any
	if instr, ok := reqMap["instructions"].(string); ok && instr != "" {
		messages = append(messages, map[string]any{"role": "system", "content": instr})
	}
	switch inp := reqMap["input"].(type) {
	case string:
		messages = append(messages, map[string]any{"role": "user", "content": inp})
	case []any:
		messages = append(messages, inp...)
	}

	reqModel, _ := reqMap["model"].(string)
	servedList := gs.getServedModelList()
	if reqModel == "" {
		reqModel = servedList[0]
	} else if !gs.isServedModel(reqModel) {
		writeOpenAIError(w, http.StatusNotFound,
			fmt.Sprintf("The model %q does not exist (served models: %s).", reqModel, strings.Join(servedList, ", ")),
			"invalid_request_error", "model", "model_not_found")
		return
	}

	chatReq := map[string]any{
		"model":       CanonicalUpstreamModel,
		"messages":    messages,
		"stream":      false,
		"temperature": reqMap["temperature"],
	}
	if maxOut, ok := reqMap["max_output_tokens"]; ok {
		chatReq["max_tokens"] = maxOut
	}
	if tools, ok := reqMap["tools"]; ok {
		chatReq["tools"] = tools
	}
	if reasoning, ok := reqMap["reasoning"]; ok {
		chatReq["reasoning"] = reasoning
	}

	reasoningCfg, err := NormalizeReasoningAndMessages(chatReq)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "reasoning", "invalid_value")
		return
	}
	chatBytes, _ := json.Marshal(chatReq)

	upReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(chatBytes))
	upReq.Header.Set("Content-Type", "application/json")
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	var rec FakeResponseWriter
	rec.header = make(http.Header)
	gs.handleNonStreamingChatResponse(&rec, resp.Body, reasoningCfg, "")

	var chatResp map[string]any
	_ = json.Unmarshal(rec.buf.Bytes(), &chatResp)

	var outputItems []any
	if choices, ok := chatResp["choices"].([]any); ok && len(choices) > 0 {
		if ch, ok := choices[0].(map[string]any); ok {
			if msg, ok := ch["message"].(map[string]any); ok {
				if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
					outputItems = append(outputItems, map[string]any{
						"type": "reasoning",
						"id":   "rs_" + strconv.FormatInt(time.Now().UnixNano(), 10),
						"summary": []any{
							map[string]any{"type": "summary_text", "text": rc},
						},
					})
				}
				if c, ok := msg["content"].(string); ok && c != "" {
					outputItems = append(outputItems, map[string]any{
						"type":   "message",
						"id":     "msg_" + strconv.FormatInt(time.Now().UnixNano(), 10),
						"status": "completed",
						"role":   "assistant",
						"content": []any{
							map[string]any{"type": "output_text", "text": c},
						},
					})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok {
					for _, tcRaw := range tcs {
						if tc, ok := tcRaw.(map[string]any); ok {
							fn, _ := tc["function"].(map[string]any)
							outputItems = append(outputItems, map[string]any{
								"type":      "function_call",
								"id":        tc["id"],
								"call_id":   tc["id"],
								"name":      fn["name"],
								"arguments": fn["arguments"],
								"status":    "completed",
							})
						}
					}
				}
			}
		}
	}

	out := map[string]any{
		"id":         "resp_" + strconv.FormatInt(time.Now().UnixNano(), 10),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      reqModel,
		"output":     outputItems,
		"usage":      chatResp["usage"],
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

type FakeResponseWriter struct {
	header http.Header
	buf    bytes.Buffer
	status int
}

func (f *FakeResponseWriter) Header() http.Header         { return f.header }
func (f *FakeResponseWriter) Write(b []byte) (int, error) { return f.buf.Write(b) }
func (f *FakeResponseWriter) WriteHeader(statusCode int)  { f.status = statusCode }

// ============================================================================
// 6. Anthropic `/v1/messages` & `/v1/messages/count_tokens` Compatibility Endpoint
// ============================================================================

func (gs *GatewayServer) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	gs.totalRequests.Add(1)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var anthReq map[string]any
	if err := json.Unmarshal(bodyBytes, &anthReq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error())
		return
	}

	reqModel, _ := anthReq["model"].(string)
	if !gs.isServedModel(reqModel) {
		servedList := gs.getServedModelList()
		writeAnthropicError(w, http.StatusNotFound, "not_found_error",
			fmt.Sprintf("model %q is not served by this engine (served models: %s)", reqModel, strings.Join(servedList, ", ")))
		return
	}

	var oaMsgs []map[string]any
	if sysPrompt := extractTextContent(anthReq["system"]); sysPrompt != "" {
		oaMsgs = append(oaMsgs, map[string]any{"role": "system", "content": sysPrompt})
	}

	if rawMsgs, ok := anthReq["messages"].([]any); ok {
		for _, rm := range rawMsgs {
			m, ok := rm.(map[string]any)
			if !ok {
				continue
			}
			role, _ := m["role"].(string)
			switch content := m["content"].(type) {
			case string:
				oaMsgs = append(oaMsgs, map[string]any{"role": role, "content": content})
			case []any:
				var textParts []string
				var toolCalls []any
				var toolResults []map[string]any
				var thinkingText string
				for _, blkRaw := range content {
					blk, ok := blkRaw.(map[string]any)
					if !ok {
						continue
					}
					bType, _ := blk["type"].(string)
					switch bType {
					case "text":
						if t, ok := blk["text"].(string); ok {
							textParts = append(textParts, t)
						}
					case "thinking":
						if t, ok := blk["thinking"].(string); ok {
							thinkingText = t
						}
					case "tool_use":
						inputBytes, _ := json.Marshal(blk["input"])
						toolCalls = append(toolCalls, map[string]any{
							"id":   blk["id"],
							"type": "function",
							"function": map[string]any{
								"name":      blk["name"],
								"arguments": string(inputBytes),
							},
						})
					case "tool_result":
						toolResults = append(toolResults, map[string]any{
							"role":         "tool",
							"tool_call_id": blk["tool_use_id"],
							"content":      extractTextContent(blk["content"]),
						})
					}
				}
				for _, tr := range toolResults {
					oaMsgs = append(oaMsgs, tr)
				}
				if len(textParts) > 0 || len(toolCalls) > 0 || thinkingText != "" {
					msgObj := map[string]any{
						"role":    role,
						"content": strings.Join(textParts, "\n"),
					}
					if thinkingText != "" {
						msgObj["reasoning_content"] = thinkingText
					}
					if len(toolCalls) > 0 {
						msgObj["tool_calls"] = toolCalls
					}
					oaMsgs = append(oaMsgs, msgObj)
				}
			}
		}
	}

	var oaTools []any
	if rawTools, ok := anthReq["tools"].([]any); ok {
		for _, rt := range rawTools {
			if tm, ok := rt.(map[string]any); ok {
				oaTools = append(oaTools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        tm["name"],
						"description": tm["description"],
						"parameters":  tm["input_schema"],
					},
				})
			}
		}
	}

	if strings.HasSuffix(r.URL.Path, "/count_tokens") {
		estTokens := len(EstimateTokens(string(bodyBytes)))
		if estTokens < 1 {
			estTokens = 1
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": estTokens})
		return
	}

	enableThinking := false
	reasoningEffort := "low"
	upstreamEffort := "low"
	if defEff, ok := gs.getModelDefaultEffort(reqModel); ok {
		if eff, upEff, en, valid := parseKnownEffort(defEff); valid {
			reasoningEffort = eff
			upstreamEffort = upEff
			enableThinking = en
		}
	}

	var rawEffort string
	if oc, ok := anthReq["output_config"].(map[string]any); ok {
		if effStr, ok := oc["effort"].(string); ok && effStr != "" {
			rawEffort = effStr
		}
	}
	if rawEffort == "" {
		if effStr, ok := anthReq["reasoning_effort"].(string); ok && effStr != "" {
			rawEffort = effStr
		} else if effStr, ok := anthReq["effort"].(string); ok && effStr != "" {
			rawEffort = effStr
		}
	}
	if rawEffort != "" {
		eff, upEff, en, valid := parseKnownEffort(rawEffort)
		if !valid {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("unsupported effort level %q (supported values: none, low, medium, high, max, xhigh)", rawEffort))
			return
		}
		reasoningEffort = eff
		upstreamEffort = upEff
		enableThinking = en
	}

	if th, ok := anthReq["thinking"].(map[string]any); ok {
		tType, _ := th["type"].(string)
		switch strings.ToLower(strings.TrimSpace(tType)) {
		case "disabled", "none", "off":
			enableThinking = false
			reasoningEffort = "low"
			upstreamEffort = "low"
		case "enabled", "adaptive":
			enableThinking = true
			if rawEffort == "" {
				budget, hasBudget := th["budget_tokens"].(float64)
				if !hasBudget || budget <= 0 {
					if defEff, ok := gs.getModelDefaultEffort(reqModel); ok {
						if eff, upEff, _, valid := parseKnownEffort(defEff); valid {
							reasoningEffort = eff
							upstreamEffort = upEff
						}
					} else {
						reasoningEffort = "max"
						upstreamEffort = "xhigh"
					}
				} else {
					switch {
					case budget <= 512:
						reasoningEffort = "low"
						upstreamEffort = "low"
					case budget <= 2048:
						reasoningEffort = "medium"
						upstreamEffort = "medium"
					case budget <= 8192:
						reasoningEffort = "high"
						upstreamEffort = "high"
					default:
						reasoningEffort = "max"
						upstreamEffort = "xhigh"
					}
				}
			}
		}
	}

	ctk := map[string]any{
		"enable_thinking": enableThinking,
		"thinking":        enableThinking,
		"clear_thinking":  true,
	}
	oaReq := map[string]any{
		"model":                CanonicalUpstreamModel,
		"messages":             oaMsgs,
		"max_tokens":           anthReq["max_tokens"],
		"temperature":          anthReq["temperature"],
		"stream":               false,
		"chat_template_kwargs": ctk,
	}
	if enableThinking {
		oaReq["reasoning_effort"] = upstreamEffort
		ctk["reasoning_effort"] = upstreamEffort
	}
	if oaReq["max_tokens"] == nil {
		oaReq["max_tokens"] = 4096
	}
	if len(oaTools) > 0 {
		oaReq["tools"] = oaTools
	}

	hitBlocks, hitTokens, nvmeRestored := gs.router.RouteAndPin(oaMsgs, oaTools)
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Blocks", strconv.Itoa(hitBlocks))
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Tokens", strconv.Itoa(hitTokens))
	w.Header().Set("X-Spark-NVMe-Spill-Restored", strconv.Itoa(nvmeRestored))
	w.Header().Set("X-Spark-Reasoning-Effort", reasoningEffort)

	oaBytes, _ := json.Marshal(oaReq)
	upReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(oaBytes))
	upReq.Header.Set("Content-Type", "application/json")
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	var rec FakeResponseWriter
	rec.header = make(http.Header)
	gs.handleNonStreamingChatResponse(&rec, resp.Body, ReasoningConfig{
		Effort:         reasoningEffort,
		UpstreamEffort: upstreamEffort,
		EnableThinking: enableThinking,
		ClearThinking:  true,
		HardTokenCap:   32768,
		Tools:          oaTools,
		ToolArgTypes:   ExtractToolArgTypes(oaTools),
	}, "")

	var oaResp map[string]any
	_ = json.Unmarshal(rec.buf.Bytes(), &oaResp)

	var contentBlocks []map[string]any
	stopReason := "end_turn"
	if choices, ok := oaResp["choices"].([]any); ok && len(choices) > 0 {
		if ch, ok := choices[0].(map[string]any); ok {
			if msg, ok := ch["message"].(map[string]any); ok {
				if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type":      "thinking",
						"thinking":  rc,
						"signature": "spark-dsv41-sig",
					})
				}
				if c, ok := msg["content"].(string); ok && c != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": c,
					})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok && len(tcs) > 0 {
					stopReason = "tool_use"
					for _, tcRaw := range tcs {
						if tc, ok := tcRaw.(map[string]any); ok {
							fn, _ := tc["function"].(map[string]any)
							var inputObj any = map[string]any{}
							if argStr, ok := fn["arguments"].(string); ok {
								_ = json.Unmarshal([]byte(RepairMalformedJSON(argStr)), &inputObj)
							}
							contentBlocks = append(contentBlocks, map[string]any{
								"type":  "tool_use",
								"id":    tc["id"],
								"name":  fn["name"],
								"input": inputObj,
							})
						}
					}
				}
			}
		}
	}
	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": ""})
	}

	usageMap, _ := oaResp["usage"].(map[string]any)
	inToks, _ := usageMap["prompt_tokens"].(float64)
	outToks, _ := usageMap["completion_tokens"].(float64)

	anthMsg := map[string]any{
		"id":            "msg_" + strconv.FormatInt(time.Now().UnixNano(), 10),
		"type":          "message",
		"role":          "assistant",
		"model":         anthReq["model"],
		"content":       contentBlocks,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":                int(inToks),
			"output_tokens":               int(outToks),
			"cache_read_input_tokens":     hitTokens,
			"cache_creation_input_tokens": 0,
		},
	}

	isStream, _ := anthReq["stream"].(bool)
	if !isStream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(anthMsg)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	sendEvent := func(event string, payload any) {
		b, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(b))
		if flusher != nil {
			flusher.Flush()
		}
	}

	sendEvent("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            anthMsg["id"],
			"type":          "message",
			"role":          "assistant",
			"model":         anthReq["model"],
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":                int(inToks),
				"output_tokens":               0,
				"cache_read_input_tokens":     hitTokens,
				"cache_creation_input_tokens": 0,
			},
		},
	})
	for idx, blk := range contentBlocks {
		bType, _ := blk["type"].(string)
		switch bType {
		case "text":
			sendEvent("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "text_delta", "text": blk["text"]},
			})
		case "thinking":
			sendEvent("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "thinking_delta", "thinking": blk["thinking"]},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "signature_delta", "signature": "spark-dsv41-sig"},
			})
		case "tool_use":
			inBytes, _ := json.Marshal(blk["input"])
			sendEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    blk["id"],
					"name":  blk["name"],
					"input": map[string]any{},
				},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(inBytes)},
			})
		}
		sendEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
	}
	sendEvent("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": int(outToks)},
	})
	sendEvent("message_stop", map[string]any{"type": "message_stop"})
}

func (gs *GatewayServer) handleProxyPass(w http.ResponseWriter, r *http.Request) {
	targetURL := gs.upstreamURL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	var bodyReader io.Reader
	if r.Body != nil {
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		bodyReader = bytes.NewReader(b)
	}
	upReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bodyReader)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for k, vals := range r.Header {
		for _, v := range vals {
			upReq.Header.Add(k, v)
		}
	}
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (gs *GatewayServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	upResp, err := gs.client.Get(gs.upstreamURL + "/health")
	upOK := err == nil && upResp != nil && upResp.StatusCode == http.StatusOK
	if upResp != nil && upResp.Body != nil {
		upResp.Body.Close()
	}
	status := "ok"
	code := http.StatusOK
	if !upOK {
		status = "degraded"
		code = http.StatusServiceUnavailable
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":           status,
		"engine":           "spark-dsv41",
		"upstream_healthy": upOK,
		"radix_prefix_router": map[string]any{
			"block_size_tokens":       gs.router.blockSize,
			"lookups_total":           gs.router.totalLookups.Load(),
			"block_hits_total":        gs.router.totalBlockHits.Load(),
			"token_hits_total":        gs.router.totalTokenHits.Load(),
			"pinned_blocks":           gs.router.pinnedBlocks.Load(),
			"nvme_spill_writes_total": gs.router.nvmeSpillWrites.Load(),
			"nvme_spill_reads_total":  gs.router.nvmeSpillReads.Load(),
		},
	})
}

func (gs *GatewayServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP spark_dsv41_gateway_requests_total Total requests handled by Spark-DSV41 Go Gateway\n")
	fmt.Fprintf(w, "# TYPE spark_dsv41_gateway_requests_total counter\n")
	fmt.Fprintf(w, "spark_dsv41_gateway_requests_total %d\n", gs.totalRequests.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_radix_lookups_total %d\n", gs.router.totalLookups.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_radix_block_hits_total %d\n", gs.router.totalBlockHits.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_radix_token_hits_total %d\n", gs.router.totalTokenHits.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_radix_pinned_blocks %d\n", gs.router.pinnedBlocks.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_nvme_spill_writes_total %d\n", gs.router.nvmeSpillWrites.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_nvme_spill_reads_total %d\n", gs.router.nvmeSpillReads.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_repaired_tool_calls_total %d\n", gs.repairedToolCalls.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_loop_guard_aborts_total %d\n", gs.loopGuardAborts.Load())
	fmt.Fprintf(w, "spark_dsv41_gateway_client_disconnects_total %d\n", gs.clientDisconnects.Load())

	resp, err := gs.client.Get(gs.upstreamURL + "/metrics")
	if err == nil {
		defer resp.Body.Close()
		_, _ = io.Copy(w, resp.Body)
	}
}

func (gs *GatewayServer) handleModels(w http.ResponseWriter, r *http.Request) {
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet, gs.upstreamURL+"/v1/models", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var templateModel map[string]any
	if resp, err := gs.client.Do(upReq); err == nil {
		defer resp.Body.Close()
		if body, err := io.ReadAll(resp.Body); err == nil && resp.StatusCode == http.StatusOK {
			var parsed map[string]any
			if json.Unmarshal(body, &parsed) == nil {
				if dataList, ok := parsed["data"].([]any); ok && len(dataList) > 0 {
					if firstModel, ok := dataList[0].(map[string]any); ok {
						templateModel = firstModel
					}
				}
			}
		}
	}
	if templateModel == nil {
		templateModel = map[string]any{
			"object":   "model",
			"owned_by": "spark-dsv41",
		}
	}

	buildEntry := func(id string) map[string]any {
		cp := make(map[string]any, len(templateModel)+5)
		for k, v := range templateModel {
			cp[k] = v
		}
		cp["id"] = id
		cp["root"] = "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw"
		cp["supported_reasoning_efforts"] = []string{"none", "low", "medium", "high", "max", "xhigh"}
		if defEff, ok := gs.getModelDefaultEffort(id); ok {
			cp["default_effort"] = defEff
		}
		return cp
	}

	servedList := gs.getServedModelList()
	trimmedPath := strings.TrimPrefix(strings.TrimPrefix(r.URL.Path, "/v1/models"), "/models")
	trimmedPath = strings.TrimPrefix(trimmedPath, "/")
	if trimmedPath != "" {
		if !gs.isServedModel(trimmedPath) {
			writeOpenAIError(w, http.StatusNotFound,
				fmt.Sprintf("The model %q does not exist (served models: %s).", trimmedPath, strings.Join(servedList, ", ")),
				"invalid_request_error", "model", "model_not_found")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(buildEntry(trimmedPath))
		return
	}

	outList := make([]any, 0, len(servedList))
	for _, m := range servedList {
		outList = append(outList, buildEntry(m))
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   outList,
	})
}

func main() {
	listenAddr := flag.String("listen", ":8888", "Listen address for Spark-DSV41 Go Control Plane")
	upstreamURL := flag.String("upstream", "http://127.0.0.1:8000", "Upstream internal TensorFold engine URL")
	backendURL := flag.String("backend", "", "Alias for -upstream")
	spillDir := flag.String("spill-dir", "/tmp/spark_dsv41_kv_spill", "Directory for O_DIRECT NVMe KV Spill Tier")
	blockSize := flag.Int("block-size", DefaultBlockSizeTokens, "KV cache block size in tokens")
	servedModelsFlag := flag.String("served-models", "", "Comma-separated list of served model names")
	aliasesConfigFlag := flag.String("aliases-config", "", "Path to runtime JSON model aliases config (default: config/model_aliases.json)")
	flag.Parse()
	if *backendURL != "" {
		*upstreamURL = *backendURL
	}

	rawServed := *servedModelsFlag
	if rawServed == "" {
		rawServed = os.Getenv("SERVED_MODEL_NAMES")
	}
	var servedList []string
	if rawServed != "" {
		for _, part := range strings.FieldsFunc(rawServed, func(r rune) bool { return r == ',' || r == ' ' }) {
			if part != "" {
				servedList = append(servedList, part)
			}
		}
	}

	aliasesCfgPath := *aliasesConfigFlag
	if aliasesCfgPath == "" {
		aliasesCfgPath = os.Getenv("SPARK_ALIASES_CONFIG")
	}
	if aliasesCfgPath == "" {
		if exe, err := os.Executable(); err == nil {
			aliasesCfgPath = filepath.Join(filepath.Dir(filepath.Dir(exe)), "config", "model_aliases.json")
		} else {
			aliasesCfgPath = "config/model_aliases.json"
		}
	}

	router := NewRadixPrefixRouter(*blockSize, MaxRadixBlocks, *spillDir)
	srv := NewGatewayServer(*upstreamURL, router, servedList, aliasesCfgPath)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/v1/responses", srv.handleResponsesAPI)
	mux.HandleFunc("/v1/messages", srv.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages/", srv.handleAnthropicMessages)
	mux.HandleFunc("/messages", srv.handleAnthropicMessages)
	mux.HandleFunc("/messages/", srv.handleAnthropicMessages)
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/models/", srv.handleModels)
	mux.HandleFunc("/models", srv.handleModels)
	mux.HandleFunc("/models/", srv.handleModels)
	mux.HandleFunc("/tokenize", srv.handleProxyPass)
	mux.HandleFunc("/detokenize", srv.handleProxyPass)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/metrics", srv.handleMetrics)

	log.Printf("[spark-dsv41-gateway] Starting Spark-DSV41 Go Control Plane on %s -> upstream %s (block_size=%d, spill_dir=%s, aliases_config=%s, served_models=%v)",
		*listenAddr, *upstreamURL, *blockSize, *spillDir, aliasesCfgPath, srv.getServedModelList())
	httpSrv := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 30 * time.Second,
	}
	if err := httpSrv.ListenAndServe(); err != nil {
		log.Fatalf("[spark-dsv41-gateway] server error: %v", err)
	}
}
