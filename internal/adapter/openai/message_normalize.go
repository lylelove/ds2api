package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"ds2api/internal/config"
)

func normalizeOpenAIMessagesForPrompt(raw []any, traceID string) []map[string]any {
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		msg, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		switch role {
		case "assistant":
			content := normalizeOpenAIContentForPrompt(msg["content"])
			toolCalls := formatAssistantToolCallsForPrompt(msg, traceID)
			combined := joinNonEmpty(content, toolCalls)
			if combined == "" {
				continue
			}
			out = append(out, map[string]any{
				"role":    "assistant",
				"content": combined,
			})
		case "tool", "function":
			out = append(out, map[string]any{
				"role":    "user",
				"content": formatToolResultForPrompt(msg),
			})
		case "user", "system", "developer":
			out = append(out, map[string]any{
				"role":    normalizeOpenAIRoleForPrompt(role),
				"content": normalizeOpenAIContentForPrompt(msg["content"]),
			})
		default:
			content := normalizeOpenAIContentForPrompt(msg["content"])
			if content == "" {
				continue
			}
			if role == "" {
				role = "user"
			}
			out = append(out, map[string]any{
				"role":    normalizeOpenAIRoleForPrompt(role),
				"content": content,
			})
		}
	}
	return out
}

func formatAssistantToolCallsForPrompt(msg map[string]any, traceID string) string {
	entries := make([]string, 0)
	if calls, ok := msg["tool_calls"].([]any); ok {
		for i, item := range calls {
			call, ok := item.(map[string]any)
			if !ok {
				continue
			}
			id := strings.TrimSpace(asString(call["id"]))
			if id == "" {
				id = fmt.Sprintf("call_%d", i+1)
			}
			name := strings.TrimSpace(asString(call["name"]))
			args := ""

			if fn, ok := call["function"].(map[string]any); ok {
				if name == "" {
					name = strings.TrimSpace(asString(fn["name"]))
				}
				args = normalizeOpenAIArgumentsForPrompt(fn["arguments"])
			}
			if name == "" {
				name = "unknown"
			}
			if args == "" {
				args = normalizeOpenAIArgumentsForPrompt(call["arguments"])
			}
			if args == "" {
				args = normalizeOpenAIArgumentsForPrompt(call["input"])
			}
			if args == "" {
				args = "{}"
			}
			maybeWarnSuspiciousToolHistory(traceID, id, name, args)
			entries = append(entries, fmt.Sprintf("[TOOL_CALL_HISTORY]\nstatus: already_called\norigin: assistant\nnot_user_input: true\ntool_call_id: %s\nfunction.name: %s\nfunction.arguments: %s\n[/TOOL_CALL_HISTORY]", id, name, args))
		}
	}

	if legacy, ok := msg["function_call"].(map[string]any); ok {
		name := strings.TrimSpace(asString(legacy["name"]))
		if name == "" {
			name = "unknown"
		}
		args := normalizeOpenAIArgumentsForPrompt(legacy["arguments"])
		if args == "" {
			args = "{}"
		}
		maybeWarnSuspiciousToolHistory(traceID, "call_legacy", name, args)
		entries = append(entries, fmt.Sprintf("[TOOL_CALL_HISTORY]\nstatus: already_called\norigin: assistant\nnot_user_input: true\ntool_call_id: call_legacy\nfunction.name: %s\nfunction.arguments: %s\n[/TOOL_CALL_HISTORY]", name, args))
	}

	return strings.Join(entries, "\n\n")
}

func formatToolResultForPrompt(msg map[string]any) string {
	toolCallID := strings.TrimSpace(asString(msg["tool_call_id"]))
	if toolCallID == "" {
		toolCallID = strings.TrimSpace(asString(msg["id"]))
	}
	if toolCallID == "" {
		toolCallID = "unknown"
	}

	name := strings.TrimSpace(asString(msg["name"]))
	if name == "" {
		name = "unknown"
	}

	content := normalizeOpenAIContentForPrompt(msg["content"])
	if content == "" {
		content = "null"
	}

	return fmt.Sprintf("[TOOL_RESULT_HISTORY]\nstatus: already_returned\norigin: tool_runtime\nnot_user_input: true\ntool_call_id: %s\nname: %s\ncontent: %s\n[/TOOL_RESULT_HISTORY]", toolCallID, name, content)
}

func normalizeOpenAIContentForPrompt(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			t := strings.ToLower(strings.TrimSpace(asString(m["type"])))
			if t != "text" && t != "output_text" && t != "input_text" {
				continue
			}
			if text := asString(m["text"]); text != "" {
				parts = append(parts, text)
				continue
			}
			if text := asString(m["content"]); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return marshalToPromptString(v)
	}
}

func normalizeOpenAIArgumentsForPrompt(v any) string {
	switch x := v.(type) {
	case string:
		return normalizeToolArgumentString(x)
	default:
		return marshalToPromptString(v)
	}
}

func normalizeToolArgumentString(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if looksLikeConcatenatedJSON(trimmed) {
		// Keep original payload to avoid silent argument rewrites.
		return raw
	}
	return trimmed
}

func marshalToPromptString(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return strings.TrimSpace(fmt.Sprintf("%v", v))
	}
	return string(b)
}

func normalizeOpenAIRoleForPrompt(role string) string {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "developer" {
		return "system"
	}
	return role
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func joinNonEmpty(parts ...string) string {
	nonEmpty := make([]string, 0, len(parts))
	for _, p := range parts {
		if strings.TrimSpace(p) == "" {
			continue
		}
		nonEmpty = append(nonEmpty, p)
	}
	return strings.Join(nonEmpty, "\n\n")
}

func maybeWarnSuspiciousToolHistory(traceID, callID, name, args string) {
	if !looksLikeConcatenatedJSON(args) {
		return
	}
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		traceID = "unknown"
	}
	config.Logger.Warn(
		"[openai] suspicious tool call history payload detected",
		"trace_id", traceID,
		"tool_call_id", strings.TrimSpace(callID),
		"name", strings.TrimSpace(name),
		"arguments_preview", previewToolArgs(args, 160),
	)
}

func looksLikeConcatenatedJSON(raw string) bool {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return false
	}
	if strings.Contains(trimmed, "}{") || strings.Contains(trimmed, "][") {
		return true
	}
	dec := json.NewDecoder(strings.NewReader(trimmed))
	var first any
	if err := dec.Decode(&first); err != nil {
		return false
	}
	var second any
	return dec.Decode(&second) == nil
}

func previewToolArgs(raw string, max int) string {
	trimmed := strings.TrimSpace(raw)
	if max <= 0 || len(trimmed) <= max {
		return trimmed
	}
	return trimmed[:max]
}
