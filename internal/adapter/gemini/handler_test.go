package gemini

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"ds2api/internal/auth"
)

type testGeminiConfig struct{}

func (testGeminiConfig) ModelAliases() map[string]string { return nil }

type testGeminiAuth struct {
	a   *auth.RequestAuth
	err error
}

func (m testGeminiAuth) Determine(_ *http.Request) (*auth.RequestAuth, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.a != nil {
		return m.a, nil
	}
	return &auth.RequestAuth{
		UseConfigToken: false,
		DeepSeekToken:  "direct-token",
		CallerID:       "caller:test",
		TriedAccounts:  map[string]bool{},
	}, nil
}

func (testGeminiAuth) Release(_ *auth.RequestAuth) {}

type testGeminiDS struct {
	resp *http.Response
	err  error
}

func (m testGeminiDS) CreateSession(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "session-id", nil
}

func (m testGeminiDS) GetPow(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "pow", nil
}

func (m testGeminiDS) CallCompletion(_ context.Context, _ *auth.RequestAuth, _ map[string]any, _ string, _ int) (*http.Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.resp, nil
}

func makeGeminiUpstreamResponse(lines ...string) *http.Response {
	body := strings.Join(lines, "\n")
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestGeminiRoutesRegistered(t *testing.T) {
	h := &Handler{
		Store: testGeminiConfig{},
		Auth:  testGeminiAuth{err: auth.ErrUnauthorized},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	paths := []string{
		"/v1beta/models/gemini-2.5-pro:generateContent",
		"/v1beta/models/gemini-2.5-pro:streamGenerateContent",
		"/v1/models/gemini-2.5-pro:generateContent",
		"/v1/models/gemini-2.5-pro:streamGenerateContent",
	}
	for _, path := range paths {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if rec.Code == http.StatusNotFound {
			t.Fatalf("expected route %s to be registered, got 404", path)
		}
	}
}

func TestGenerateContentReturnsFunctionCallParts(t *testing.T) {
	upstream := makeGeminiUpstreamResponse(
		`data: {"p":"response/content","v":"{\"tool_calls\":[{\"name\":\"eval_javascript\",\"input\":{\"code\":\"1+1\"}}]}"}`,
		`data: [DONE]`,
	)
	h := &Handler{
		Store: testGeminiConfig{},
		Auth:  testGeminiAuth{},
		DS:    testGeminiDS{resp: upstream},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{
		"contents":[{"role":"user","parts":[{"text":"call tool"}]}],
		"tools":[{"functionDeclarations":[{"name":"eval_javascript","description":"eval","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}]}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-pro:generateContent", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer direct-token")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	candidates, _ := out["candidates"].([]any)
	if len(candidates) == 0 {
		t.Fatalf("expected non-empty candidates: %#v", out)
	}
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	parts, _ := content["parts"].([]any)
	if len(parts) == 0 {
		t.Fatalf("expected non-empty parts: %#v", content)
	}
	part0, _ := parts[0].(map[string]any)
	functionCall, _ := part0["functionCall"].(map[string]any)
	if functionCall["name"] != "eval_javascript" {
		t.Fatalf("expected functionCall name eval_javascript, got %#v", functionCall)
	}
}

func TestGenerateContentMixedToolSnippetAlsoTriggersFunctionCall(t *testing.T) {
	upstream := makeGeminiUpstreamResponse(
		`data: {"p":"response/content","v":"我来调用工具\n{\"tool_calls\":[{\"name\":\"eval_javascript\",\"input\":{\"code\":\"1+1\"}}]}"}`,
		`data: [DONE]`,
	)
	h := &Handler{Store: testGeminiConfig{}, Auth: testGeminiAuth{}, DS: testGeminiDS{resp: upstream}}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{
		"contents":[{"role":"user","parts":[{"text":"call tool"}]}],
		"tools":[{"functionDeclarations":[{"name":"eval_javascript","description":"eval","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}]}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-pro:generateContent", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer direct-token")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	candidates, _ := out["candidates"].([]any)
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	parts, _ := content["parts"].([]any)
	part0, _ := parts[0].(map[string]any)
	functionCall, _ := part0["functionCall"].(map[string]any)
	if functionCall["name"] != "eval_javascript" {
		t.Fatalf("expected functionCall name eval_javascript for mixed snippet, got %#v", functionCall)
	}
}

func TestStreamGenerateContentEmitsSSE(t *testing.T) {
	upstream := makeGeminiUpstreamResponse(
		`data: {"p":"response/content","v":"hello "}`,
		`data: {"p":"response/content","v":"world"}`,
		`data: [DONE]`,
	)
	h := &Handler{
		Store: testGeminiConfig{},
		Auth:  testGeminiAuth{},
		DS:    testGeminiDS{resp: upstream},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/models/gemini-2.5-pro:streamGenerateContent?alt=sse", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer direct-token")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "data: ") {
		t.Fatalf("expected SSE data frames, got body=%s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"finishReason":"STOP"`) {
		t.Fatalf("expected stream finish frame, got body=%s", rec.Body.String())
	}

	frames := extractGeminiSSEFrames(t, rec.Body.String())
	if len(frames) == 0 {
		t.Fatalf("expected non-empty sse frames, body=%s", rec.Body.String())
	}
	last := frames[len(frames)-1]
	candidates, _ := last["candidates"].([]any)
	if len(candidates) == 0 {
		t.Fatalf("expected finish frame candidates, got %#v", last)
	}
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	if content == nil {
		t.Fatalf("expected non-null content in finish frame, got %#v", c0)
	}
	parts, _ := content["parts"].([]any)
	if len(parts) == 0 {
		t.Fatalf("expected non-empty parts in finish frame content, got %#v", content)
	}
}

func extractGeminiSSEFrames(t *testing.T, body string) []map[string]any {
	t.Helper()
	scanner := bufio.NewScanner(strings.NewReader(body))
	out := make([]map[string]any, 0, 4)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if raw == "" {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(raw), &frame); err != nil {
			continue
		}
		out = append(out, frame)
	}
	return out
}
