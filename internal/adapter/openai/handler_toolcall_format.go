package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"

	"ds2api/internal/util"
)

func injectToolPrompt(messages []map[string]any, tools []any, policy util.ToolChoicePolicy) ([]map[string]any, []string) {
	if policy.IsNone() {
		return messages, nil
	}
	toolSchemas := make([]string, 0, len(tools))
	names := make([]string, 0, len(tools))
	isAllowed := func(name string) bool {
		if strings.TrimSpace(name) == "" {
			return false
		}
		if len(policy.Allowed) == 0 {
			return true
		}
		_, ok := policy.Allowed[name]
		return ok
	}

	for _, t := range tools {
		tool, ok := t.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := tool["function"].(map[string]any)
		if len(fn) == 0 {
			fn = tool
		}
		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)
		schema, _ := fn["parameters"].(map[string]any)
		name = strings.TrimSpace(name)
		if !isAllowed(name) {
			continue
		}
		names = append(names, name)
		if desc == "" {
			desc = "No description available"
		}
		b, _ := json.Marshal(schema)
		toolSchemas = append(toolSchemas, fmt.Sprintf("Tool: %s\nDescription: %s\nParameters: %s", name, desc, string(b)))
	}
	if len(toolSchemas) == 0 {
		return messages, names
	}
	toolPrompt := "You have access to these tools:\n\n" + strings.Join(toolSchemas, "\n\n") + "\n\n" + buildToolCallInstructions(names)
	if policy.Mode == util.ToolChoiceRequired {
		toolPrompt += "\n7) For this response, you MUST call at least one tool from the allowed list."
	}
	if policy.Mode == util.ToolChoiceForced && strings.TrimSpace(policy.ForcedName) != "" {
		toolPrompt += "\n7) For this response, you MUST call exactly this tool name: " + strings.TrimSpace(policy.ForcedName)
		toolPrompt += "\n8) Do not call any other tool."
	}

	for i := range messages {
		if messages[i]["role"] == "system" {
			old, _ := messages[i]["content"].(string)
			messages[i]["content"] = strings.TrimSpace(old + "\n\n" + toolPrompt)
			return messages, names
		}
	}
	messages = append([]map[string]any{{"role": "system", "content": toolPrompt}}, messages...)
	return messages, names
}

// buildToolCallInstructions generates the tool-calling instruction block
// with attention-optimized structure: rules → negative examples → positive
// examples (using real tool names) → anchor. This ordering exploits the
// transformer's recency bias: the last tokens before generation starts
// carry the strongest influence on the model's first output tokens.
func buildToolCallInstructions(toolNames []string) string {
	// Pick real tool names for examples; fall back to generic names.
	ex1 := "read_file"
	ex2 := "write_to_file"
	ex3 := "ask_followup_question"
	used := map[string]bool{}
	for _, n := range toolNames {
		switch {
		case !used["ex1"] && (n == "read_file" || n == "list_files" || n == "search_files"):
			ex1 = n
			used["ex1"] = true
		case !used["ex2"] && (n == "write_to_file" || n == "apply_diff" || n == "execute_command"):
			ex2 = n
			used["ex2"] = true
		case !used["ex3"] && (n == "ask_followup_question" || n == "attempt_completion" || n == "update_todo_list"):
			ex3 = n
			used["ex3"] = true
		}
	}

	return `TOOL CALL FORMAT — FOLLOW EXACTLY:

When calling tools, emit ONLY raw XML. No text before, no text after, no markdown fences.

<tool_calls>
  <tool_call>
    <tool_name>TOOL_NAME_HERE</tool_name>
    <parameters>{"key":"value"}</parameters>
  </tool_call>
</tool_calls>

RULES:
1) Output ONLY the XML above when calling tools. Do NOT mix tool XML with regular text.
2) <parameters> MUST contain a strict JSON object. All JSON keys and strings use double quotes.
3) Multiple tools → multiple <tool_call> blocks inside ONE <tool_calls> root.
4) Do NOT wrap the XML in markdown code fences (no triple backticks).
5) After receiving a tool result, use it directly. Only call another tool if the result is insufficient.
6) If you want to say something AND call a tool, output text first, then the XML block on its own.

❌ WRONG — Do NOT do these:
Wrong 1 — mixed text and XML:
  I'll read the file for you. <tool_calls><tool_call>...
Wrong 2 — code fence wrapping:
  ` + "```xml\n  <tool_calls>...\n  ```" + `
Wrong 3 — missing <tool_calls> wrapper:
  <tool_call><tool_name>` + ex1 + `</tool_name><parameters>{}</parameters></tool_call>

✅ CORRECT EXAMPLES:

Example A — Single tool:
<tool_calls>
  <tool_call>
    <tool_name>` + ex1 + `</tool_name>
    <parameters>{"path":"src/main.go"}</parameters>
  </tool_call>
</tool_calls>

Example B — Two tools in parallel:
<tool_calls>
  <tool_call>
    <tool_name>` + ex1 + `</tool_name>
    <parameters>{"path":"config.json"}</parameters>
  </tool_call>
  <tool_call>
    <tool_name>` + ex2 + `</tool_name>
    <parameters>{"path":"output.txt","content":"Hello world"}</parameters>
  </tool_call>
</tool_calls>

Example C — Tool with complex JSON parameters (newlines in values use \n):
<tool_calls>
  <tool_call>
    <tool_name>` + ex3 + `</tool_name>
    <parameters>{"question":"Which approach do you prefer?","follow_up":[{"text":"Option A"},{"text":"Option B"}]}</parameters>
  </tool_call>
</tool_calls>

Remember: Output ONLY the <tool_calls>...</tool_calls> XML block when calling tools.`
}
func formatIncrementalStreamToolCallDeltas(deltas []toolCallDelta, ids map[int]string) []map[string]any {
	if len(deltas) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(deltas))
	for _, d := range deltas {
		if d.Name == "" && d.Arguments == "" {
			continue
		}
		callID, ok := ids[d.Index]
		if !ok || callID == "" {
			callID = "call_" + strings.ReplaceAll(uuid.NewString(), "-", "")
			ids[d.Index] = callID
		}
		item := map[string]any{
			"index": d.Index,
			"id":    callID,
			"type":  "function",
		}
		fn := map[string]any{}
		if d.Name != "" {
			fn["name"] = d.Name
		}
		if d.Arguments != "" {
			fn["arguments"] = d.Arguments
		}
		if len(fn) > 0 {
			item["function"] = fn
		}
		out = append(out, item)
	}
	return out
}

func filterIncrementalToolCallDeltasByAllowed(deltas []toolCallDelta, allowedNames []string, seenNames map[int]string) []toolCallDelta {
	if len(deltas) == 0 {
		return nil
	}
	out := make([]toolCallDelta, 0, len(deltas))
	for _, d := range deltas {
		if d.Name != "" {
			if seenNames != nil {
				seenNames[d.Index] = d.Name
			}
			out = append(out, d)
			continue
		}
		if seenNames == nil {
			out = append(out, d)
			continue
		}
		name := strings.TrimSpace(seenNames[d.Index])
		if name == "" {
			continue
		}
		out = append(out, d)
	}
	return out
}

func formatFinalStreamToolCallsWithStableIDs(calls []util.ParsedToolCall, ids map[int]string) []map[string]any {
	if len(calls) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(calls))
	for i, c := range calls {
		callID := ""
		if ids != nil {
			callID = strings.TrimSpace(ids[i])
		}
		if callID == "" {
			callID = "call_" + strings.ReplaceAll(uuid.NewString(), "-", "")
			if ids != nil {
				ids[i] = callID
			}
		}
		args, _ := json.Marshal(c.Input)
		out = append(out, map[string]any{
			"index": i,
			"id":    callID,
			"type":  "function",
			"function": map[string]any{
				"name":      c.Name,
				"arguments": string(args),
			},
		})
	}
	return out
}
