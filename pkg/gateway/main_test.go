package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParseDSMLToolCallsAndSchemaCoercion(t *testing.T) {
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "deploy_service",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"port":    map[string]any{"type": "integer"},
						"tag":     map[string]any{"type": "string"},
						"dry_run": map[string]any{"type": "boolean"},
					},
				},
			},
		},
	}
	schemaMap := ExtractToolArgTypes(tools)

	rawDSML := `I will deploy the service now.
<｜DSML｜function_calls>
<｜DSML｜invoke name="deploy_service">
<｜DSML｜parameter name="port" string="false">8080</｜DSML｜parameter>
<｜DSML｜parameter name="tag" string="true">8080</｜DSML｜parameter>
<｜DSML｜parameter name="dry_run" string="false">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>`

	calls, cleaned := ParseDSMLAndXMLToolCalls(rawDSML, 0, schemaMap)
	if len(calls) != 1 {
		t.Fatalf("expected 1 DSML tool call, got %d", len(calls))
	}
	if calls[0].Name != "deploy_service" {
		t.Fatalf("expected function name deploy_service, got %q", calls[0].Name)
	}
	if cleaned != "I will deploy the service now.\n" {
		t.Fatalf("unexpected cleaned text: %q", cleaned)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &args); err != nil {
		t.Fatalf("failed to unmarshal arguments: %v", err)
	}
	if portVal, ok := args["port"].(float64); !ok || int(portVal) != 8080 {
		t.Fatalf("expected port=8080 (number), got %#v", args["port"])
	}
	if tagVal, ok := args["tag"].(string); !ok || tagVal != "8080" {
		t.Fatalf("expected tag=\"8080\" (string), got %#v", args["tag"])
	}
	if dryVal, ok := args["dry_run"].(bool); !ok || !dryVal {
		t.Fatalf("expected dry_run=true (bool), got %#v", args["dry_run"])
	}
}

func TestNormalizeReasoningEffortMaxToXHigh(t *testing.T) {
	req := map[string]any{
		"model":            CanonicalUpstreamModel,
		"reasoning_effort": "max",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hello"},
			map[string]any{"role": "assistant", "content": "<think>old thought</think>Hi!", "reasoning_content": "old thought"},
			map[string]any{"role": "user", "content": "Follow up"},
		},
	}
	cfg, err := NormalizeReasoningAndMessages(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Effort != "max" || cfg.UpstreamEffort != "xhigh" || !cfg.EnableThinking {
		t.Fatalf("expected Effort=max, UpstreamEffort=xhigh, EnableThinking=true; got %+v", cfg)
	}
	if req["reasoning_effort"] != "xhigh" {
		t.Fatalf("expected upstream req[reasoning_effort]=xhigh, got %#v", req["reasoning_effort"])
	}
	msgs := req["messages"].([]any)
	firstAsst := msgs[1].(map[string]any)
	if _, hasRC := firstAsst["reasoning_content"]; hasRC {
		t.Fatalf("expected historical reasoning_content to be stripped on earlier turns")
	}
	if firstAsst["content"] != "Hi!" {
		t.Fatalf("expected <think> tag stripped from historical assistant content, got %q", firstAsst["content"])
	}
}

func TestAnthropicMessagesAndHotReloadAliases(t *testing.T) {
	var capturedUpstream map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			b, _ := os.ReadFile("/dev/null")
			_ = b
			_ = json.NewDecoder(r.Body).Decode(&capturedUpstream)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "chatcmpl-test",
				"object": "chat.completion",
				"model":  CanonicalUpstreamModel,
				"choices": []any{
					map[string]any{
						"index": 0,
						"message": map[string]any{
							"role":              "assistant",
							"reasoning_content": "step-by-step analysis",
							"content":           "Ready.",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]any{
					"prompt_tokens":     42,
					"completion_tokens": 10,
				},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	tmpDir := t.TempDir()
	aliasFile := filepath.Join(tmpDir, "model_aliases.json")
	_ = os.WriteFile(aliasFile, []byte(`{
		"models": ["DeepSeek-V4.1-Flash-1017"],
		"aliases": {
			"claude-opus-4-6": {"effort": "max"}
		}
	}`), 0644)

	router := NewRadixPrefixRouter(256, 1024, filepath.Join(tmpDir, "spill"))
	srv := NewGatewayServer(upstream.URL, router, []string{CanonicalUpstreamModel}, aliasFile)

	anthBody, _ := json.Marshal(map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 1024,
		"messages": []any{
			map[string]any{"role": "user", "content": "Test Claude CLI"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(anthBody))
	rec := httptest.NewRecorder()
	srv.handleAnthropicMessages(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /v1/messages, got %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("X-Spark-Reasoning-Effort") != "max" {
		t.Fatalf("expected X-Spark-Reasoning-Effort=max, got %q", rec.Header().Get("X-Spark-Reasoning-Effort"))
	}
	if capturedUpstream["reasoning_effort"] != "xhigh" {
		t.Fatalf("expected upstream reasoning_effort=xhigh for claude-opus-4-6 (max), got %#v", capturedUpstream["reasoning_effort"])
	}

	// Verify hot-reload of aliases file
	time.Sleep(15 * time.Millisecond)
	_ = os.WriteFile(aliasFile, []byte(`{
		"models": ["DeepSeek-V4.1-Flash-1017"],
		"aliases": {
			"dsv41-custom-fast": "low"
		}
	}`), 0644)

	if !srv.isServedModel("dsv41-custom-fast") {
		t.Fatalf("expected hot-reloaded alias dsv41-custom-fast to be recognized")
	}
}
