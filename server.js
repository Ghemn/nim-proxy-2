```js
// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const GLM_REASONING_EFFORT = 'low';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// NVIDIA NIM API configuration
const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Reasoning display toggle
const SHOW_REASONING = false;

// Thinking mode toggle
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'llama': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'moonshotai/kimi-k2.5',
  'kimi-k2': 'moonshotai/kimi-k2-instruct',
  'deepseek-uncensored':
    'nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored',
  'hermes-3-llama-3.1':
    'nousresearch/hermes-3-llama-3.1-405b:free',
  'deepseek-v4.1': 'deepseek-ai/deepseek-v4.1-flash',
  'llama-3.3': 'meta-llama/Llama-3.3-70B-Instruct',
  'qwen': 'Qwen/Qwen3-32B',
  'glm': 'z-ai/glm-5.2',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-5.3-flash': 'z-ai/glm-5.3-flash',
  'deepseek-v4new': 'deepseek-ai/deepseek-v4-pro-0813',
  'kimik3': 'moonshotai/kimi-k3',
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nemotron-ultra': 'nvidia/nemotron-3-ultra-550b-a55b'
};

// ============================================================
// MODEL LIST
// ============================================================

function getModelList() {
  const created = Math.floor(Date.now() / 1000);

  return Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created,
    owned_by: 'nvidia-nim-proxy'
  }));
}

function sendModelList(req, res) {
  const models = getModelList();

  console.log(
    `[MODEL LIST] ${req.method} ${req.originalUrl} -> ${models.length} models`
  );

  res.status(200);

  res.set({
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control':
      'no-store, no-cache, must-revalidate, proxy-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });

  res.json({
    object: 'list',
    data: models
  });
}

app.get('/v1/models', sendModelList);
app.get('/models', sendModelList);

app.options('/v1/models', (req, res) => {
  res.sendStatus(204);
});

app.options('/models', (req, res) => {
  res.sendStatus(204);
});

// ============================================================
// ROOT
// ============================================================

app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '1.2.0',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions',
      completions: '/v1/completions'
    }
  });
});

// ============================================================
// HEALTH
// ============================================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_configured: !!NIM_API_KEY
  });
});

// ============================================================
// MODEL RESOLUTION
// ============================================================

async function resolveModel(model) {
  let nimModel = MODEL_MAPPING[model];

  console.log(
    `Model mapping: ${model} -> ${nimModel || 'trying fallback'}`
  );

  if (nimModel) {
    return nimModel;
  }

  try {
    const testResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model,
        messages: [{ role: 'user', content: 'test' }],
        max_tokens: 1
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        validateStatus: status => status < 500
      }
    );

    if (testResponse.status >= 200 && testResponse.status < 300) {
      nimModel = model;
      console.log(`Model ${model} is directly supported by NIM`);
    }
  } catch (e) {
    console.log('Model test failed, using fallback logic');
  }

  if (!nimModel) {
    const modelLower = model.toLowerCase();

    if (
      modelLower.includes('gpt-4') ||
      modelLower.includes('claude-opus') ||
      modelLower.includes('405b')
    ) {
      nimModel = 'meta/llama-3.1-405b-instruct';
    } else if (
      modelLower.includes('claude') ||
      modelLower.includes('gemini') ||
      modelLower.includes('70b')
    ) {
      nimModel = 'meta/llama-3.1-70b-instruct';
    } else {
      nimModel = 'meta/llama-3.1-8b-instruct';
    }

    console.log(`Using fallback model: ${nimModel}`);
  }

  return nimModel;
}

// ============================================================
// REQUEST PARAMETER FORWARDING
// ============================================================
//
// The proxy intentionally does NOT maintain a sampler whitelist.
//
// Any parameter supplied by SillyTavern is forwarded to NIM unless
// it is one of the fields that belongs specifically to the
// OpenAI/Text Completion interface and must be transformed.
//
// This means experimental parameters are not silently discarded.
//
// If NVIDIA/NIM supports a parameter -> it can use it.
// If NVIDIA/NIM ignores a parameter -> harmless.
// If NVIDIA/NIM rejects a parameter -> the actual NIM error is
// returned to SillyTavern and logged by the proxy.
//
// This is deliberate.
// ============================================================

const CHAT_EXCLUDED_FIELDS = new Set([
  // The proxy resolves the friendly alias itself.
  'model',

  // Chat messages are already the correct NIM format.
  'messages'
]);

const TEXT_EXCLUDED_FIELDS = new Set([
  // The proxy resolves the friendly alias itself.
  'model',

  // Text Completion uses "prompt"; NIM Chat Completion uses
  // "messages". The adapter transforms it.
  'prompt',

  // We explicitly control the NIM message format.
  'messages'
]);

function forwardParameters(source, target, excludedFields) {
  for (const [key, value] of Object.entries(source)) {
    if (excludedFields.has(key)) {
      continue;
    }

    target[key] = value;
  }

  return target;
}

// ============================================================
// MODEL-SPECIFIC PARAMETERS
// ============================================================

function applyModelSpecificParameters(nimRequest, nimModel) {
  // K3:
  //
  // We intentionally DO NOT remove or override sampler parameters
  // here. If ST sends top_p, top_k, min_p, repetition_penalty,
  // presence_penalty, etc., they remain in the request.
  //
  // This allows experimentation and lets NVIDIA/NIM determine
  // whether a parameter is supported, ignored, or rejected.

  // Existing GLM-5.3 / GLM-5.3-Flash reasoning configuration.
  if (
    nimModel === 'z-ai/glm-5.3' ||
    nimModel === 'z-ai/glm-5.3-flash'
  ) {
    /*
     * Preserve an explicitly supplied reasoning_effort.
     *
     * If SillyTavern supplies one, use it.
     * Otherwise retain the proxy's existing default.
     */
    if (nimRequest.reasoning_effort === undefined) {
      nimRequest.reasoning_effort = GLM_REASONING_EFFORT;
    }
  }

  // Existing thinking-mode support for other models.
  if (
    ENABLE_THINKING_MODE &&
    nimModel !== 'z-ai/glm-5.3' &&
    nimModel !== 'z-ai/glm-5.3-flash'
  ) {
    /*
     * Don't overwrite an existing extra_body supplied by ST.
     * If one doesn't exist, create the existing thinking config.
     */
    if (!nimRequest.extra_body) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true
        }
      };
    }
  }
}

// ============================================================
// NIM ERROR RESPONSE
// ============================================================

function handleNimResponseError(response, res, stream) {
  console.error(
    'NVIDIA API error status:',
    response.status
  );

  console.error(
    'NVIDIA API error headers:',
    response.headers
  );

  if (stream && response.data) {
    let errorBody = '';

    response.data.on('data', chunk => {
      errorBody += chunk.toString();
    });

    response.data.on('end', () => {
      console.error(
        'NVIDIA API error body:',
        errorBody
      );

      try {
        const parsedError = JSON.parse(errorBody);

        res.status(response.status).json({
          error: {
            message:
              parsedError?.error?.message ||
              'NVIDIA API request failed',
            type: 'invalid_request_error',
            code: response.status,
            details: parsedError
          }
        });
      } catch {
        res.status(response.status).json({
          error: {
            message:
              errorBody ||
              'NVIDIA API request failed',
            type: 'invalid_request_error',
            code: response.status
          }
        });
      }
    });

    response.data.on('error', err => {
      console.error(
        'Error reading NVIDIA error response:',
        err
      );

      if (!res.headersSent) {
        res.status(response.status).json({
          error: {
            message:
              'NVIDIA API request failed',
            type: 'invalid_request_error',
            code: response.status
          }
        });
      }
    });

    return;
  }

  console.error(
    'NVIDIA API error body:',
    response.data
  );

  return res.status(response.status).json({
    error: {
      message:
        response.data?.error?.message ||
        'NVIDIA API request failed',
      type: 'invalid_request_error',
      code: response.status,
      details: response.data
    }
  });
}

// ============================================================
// STREAMING RESPONSE
// ============================================================

function streamNimResponse(
  response,
  res,
  outputFormat
) {
  res.setHeader(
    'Content-Type',
    'text/event-stream'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  let buffer = '';
  let reasoningStarted = false;

  response.data.on('data', chunk => {
    buffer += chunk.toString();

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    lines.forEach(line => {
      if (!line.startsWith('data: ')) {
        return;
      }

      if (line.includes('[DONE]')) {
        res.write('data: [DONE]\n\n');
        return;
      }

      try {
        const data = JSON.parse(
          line.slice(6)
        );

        const delta =
          data.choices?.[0]?.delta;

        if (!delta) {
          return;
        }

        const reasoning =
          delta.reasoning_content;

        const content =
          delta.content;

        let outputContent = '';

        if (SHOW_REASONING) {
          if (
            reasoning &&
            !reasoningStarted
          ) {
            outputContent =
              '<think>\n' +
              reasoning;

            reasoningStarted = true;
          } else if (reasoning) {
            outputContent = reasoning;
          }

          if (
            content &&
            reasoningStarted
          ) {
            outputContent +=
              '</think>\n\n' +
              content;

            reasoningStarted = false;
          } else if (content) {
            outputContent += content;
          }
        } else {
          outputContent =
            content || '';
        }

        // ------------------------------------------
        // Convert NIM Chat Completion stream into
        // SillyTavern Text Completion stream.
        // ------------------------------------------

        if (outputFormat === 'text') {
          const textResponse = {
            id:
              data.id ||
              `cmpl-${Date.now()}`,

            object:
              'text_completion',

            created:
              data.created ||
              Math.floor(
                Date.now() / 1000
              ),

            model:
              data.model || '',

            choices: [
              {
                text:
                  outputContent,

                index:
                  data.choices?.[0]
                    ?.index || 0,

                finish_reason:
                  data.choices?.[0]
                    ?.finish_reason ||
                  null
              }
            ]
          };

          res.write(
            `data: ${JSON.stringify(
              textResponse
            )}\n\n`
          );
        }

        // ------------------------------------------
        // Keep normal Chat Completion streaming.
        // ------------------------------------------

        else {
          const chatData = {
            ...data,

            choices:
              data.choices.map(
                choice => ({
                  ...choice,

                  delta: {
                    ...choice.delta,
                    content:
                      outputContent
                  }
                })
              )
          };

          if (!SHOW_REASONING) {
            delete chatData
              .choices[0]
              .delta
              .reasoning_content;
          }

          res.write(
            `data: ${JSON.stringify(
              chatData
            )}\n\n`
          );
        }
      } catch (e) {
        console.error(
          'Error parsing stream chunk:',
          e
        );
      }
    });
  });

  response.data.on('end', () => {
    console.log('Stream ended');

    res.write(
      'data: [DONE]\n\n'
    );

    res.end();
  });

  response.data.on('error', err => {
    console.error(
      'Stream error:',
      err
    );

    if (!res.headersSent) {
      res.status(500).end();
    } else {
      res.end();
    }
  });
}

// ============================================================
// CHAT RESPONSE TRANSFORMATION
// ============================================================

function createOpenAIChatResponse(
  responseData,
  requestedModel
) {
  return {
    id:
      `chatcmpl-${Date.now()}`,

    object:
      'chat.completion',

    created:
      Math.floor(Date.now() / 1000),

    model:
      requestedModel,

    choices:
      responseData.choices.map(
        choice => {
          let fullContent =
            choice.message?.content ||
            '';

          if (
            SHOW_REASONING &&
            choice.message
              ?.reasoning_content
          ) {
            fullContent =
              '<think>\n' +
              choice.message
                .reasoning_content +
              '\n</think>\n\n' +
              fullContent;
          }

          return {
            index:
              choice.index,

            message: {
              role:
                choice.message.role,

              content:
                fullContent
            },

            finish_reason:
              choice.finish_reason
          };
        }
      ),

    usage:
      responseData.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
  };
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    console.log(
      'Received chat completion request'
    );

    console.log(
      'Body:',
      JSON.stringify(
        req.body,
        null,
        2
      )
    );

    try {
      if (!NIM_API_KEY) {
        return res.status(500).json({
          error: {
            message:
              'NVIDIA API key not configured',
            type:
              'invalid_request_error',
            code: 500
          }
        });
      }

      const {
        model,
        messages,
        stream
      } = req.body;

      if (!model || !messages) {
        return res.status(400).json({
          error: {
            message:
              'Missing required fields: model and messages are required',
            type:
              'invalid_request_error',
            code: 400
          }
        });
      }

      const nimModel =
        await resolveModel(model);

      /*
       * Start with the fields required by NIM,
       * then transparently forward everything else
       * supplied by SillyTavern.
       */
      const nimRequest = {
        model: nimModel,
        messages
      };

      forwardParameters(
        req.body,
        nimRequest,
        CHAT_EXCLUDED_FIELDS
      );

      /*
       * Ensure stream reflects the actual incoming
       * request rather than being accidentally omitted.
       */
      nimRequest.stream =
        stream || false;

      applyModelSpecificParameters(
        nimRequest,
        nimModel
      );

      console.log(
        'Sending request to NVIDIA NIM:',
        JSON.stringify(
          nimRequest,
          null,
          2
        )
      );

      const response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              'Content-Type':
                'application/json'
            },

            responseType:
              stream
                ? 'stream'
                : 'json',

            validateStatus:
              () => true
          }
        );

      if (response.status >= 400) {
        return handleNimResponseError(
          response,
          res,
          stream
        );
      }

      if (stream) {
        return streamNimResponse(
          response,
          res,
          'chat'
        );
      }

      console.log(
        'Received response from NVIDIA NIM'
      );

      const openaiResponse =
        createOpenAIChatResponse(
          response.data,
          model
        );

      console.log(
        'Sending response to client'
      );

      res.json(
        openaiResponse
      );
    } catch (error) {
      console.error(
        'Proxy error:',
        error.message
      );

      console.error(
        'Error details:',
        error.response?.data ||
        error
      );

      res.status(
        error.response?.status ||
        500
      ).json({
        error: {
          message:
            error.message ||
            'Internal server error',

          type:
            'invalid_request_error',

          code:
            error.response?.status ||
            500,

          details:
            error.response?.data
        }
      });
    }
  }
);

// ============================================================
// TEXT COMPLETIONS ADAPTER
// ============================================================
//
// SillyTavern Text Completion:
//
//   POST /v1/completions
//
// NVIDIA NIM:
//
//   POST /v1/chat/completions
//
// The adapter preserves SillyTavern's entire prompt exactly.
// It does NOT attempt to reconstruct the conversation.
//
// Every other request parameter is passed through transparently.
// ============================================================

app.post(
  '/v1/completions',
  async (req, res) => {
    console.log(
      'Received text completion request'
    );

    console.log(
      'Body:',
      JSON.stringify(
        req.body,
        null,
        2
      )
    );

    try {
      if (!NIM_API_KEY) {
        return res.status(500).json({
          error: {
            message:
              'NVIDIA API key not configured',

            type:
              'invalid_request_error',

            code: 500
          }
        });
      }

      const {
        model,
        prompt,
        stream
      } = req.body;

      if (
        !model ||
        prompt === undefined
      ) {
        return res.status(400).json({
          error: {
            message:
              'Missing required fields: model and prompt are required',

            type:
              'invalid_request_error',

            code: 400
          }
        });
      }

      const nimModel =
        await resolveModel(model);

      /*
       * SillyTavern has already constructed the complete
       * Text Completion prompt.
       *
       * Preserve it exactly.
       */
      const nimRequest = {
        model: nimModel,

        messages: [
          {
            role: 'user',
            content: prompt
          }
        ]
      };

      /*
       * Transparently forward EVERY other request parameter.
       *
       * This includes sampler parameters that this proxy
       * doesn't know about.
       */
      forwardParameters(
        req.body,
        nimRequest,
        TEXT_EXCLUDED_FIELDS
      );

      /*
       * stream is forwarded through the generic parameter
       * system, but explicitly ensure it is present.
       */
      nimRequest.stream =
        stream || false;

      applyModelSpecificParameters(
        nimRequest,
        nimModel
      );

      console.log(
        'Text Completion adapter -> NIM:',
        JSON.stringify(
          nimRequest,
          null,
          2
        )
      );

      const response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              'Content-Type':
                'application/json'
            },

            responseType:
              stream
                ? 'stream'
                : 'json',

            validateStatus:
              () => true
          }
        );

      if (response.status >= 400) {
        return handleNimResponseError(
          response,
          res,
          stream
        );
      }

      if (stream) {
        return streamNimResponse(
          response,
          res,
          'text'
        );
      }

      console.log(
        'Received response from NVIDIA NIM'
      );

      let generatedText =
        response.data
          ?.choices?.[0]
          ?.message
          ?.content || '';

      if (
        SHOW_REASONING &&
        response.data
          ?.choices?.[0]
          ?.message
          ?.reasoning_content
      ) {
        generatedText =
          '<think>\n' +
          response.data
            .choices[0]
            .message
            .reasoning_content +
          '\n</think>\n\n' +
          generatedText;
      }

      const textResponse = {
        id:
          `cmpl-${Date.now()}`,

        object:
          'text_completion',

        created:
          Math.floor(
            Date.now() / 1000
          ),

        model,

        choices: [
          {
            text:
              generatedText,

            index: 0,

            logprobs: null,

            finish_reason:
              response.data
                ?.choices?.[0]
                ?.finish_reason ||
              'stop'
          }
        ],

        usage:
          response.data?.usage || {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
          }
      };

      console.log(
        'Sending Text Completion response'
      );

      res.json(
        textResponse
      );
    } catch (error) {
      console.error(
        'Text Completion adapter error:',
        error.message
      );

      console.error(
        'Error details:',
        error.response?.data ||
        error
      );

      res.status(
        error.response?.status ||
        500
      ).json({
        error: {
          message:
            error.message ||
            'Internal server error',

          type:
            'invalid_request_error',

          code:
            error.response?.status ||
            500,

          details:
            error.response?.data
        }
      });
    }
  }
);

// ============================================================
// CATCH-ALL
// ============================================================

app.all('*', (req, res) => {
  console.log(
    `404: ${req.method} ${req.path} not found`
  );

  res.status(404).json({
    error: {
      message:
        `Endpoint ${req.method} ${req.path} not found`,

      type:
        'invalid_request_error',

      code: 404
    }
  });
});

// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  '0.0.0.0',
  () => {
    console.log(
      '========================================'
    );

    console.log(
      `OpenAI to NVIDIA NIM Proxy running on port ${PORT}`
    );

    console.log(
      `Health check: http://localhost:${PORT}/health`
    );

    console.log(
      `Models: http://localhost:${PORT}/v1/models`
    );

    console.log(
      `Chat: POST http://localhost:${PORT}/v1/chat/completions`
    );

    console.log(
      `Text: POST http://localhost:${PORT}/v1/completions`
    );

    console.log(
      `Reasoning display: ${
        SHOW_REASONING
          ? 'ENABLED'
          : 'DISABLED'
      }`
    );

    console.log(
      `Thinking mode: ${
        ENABLE_THINKING_MODE
          ? 'ENABLED'
          : 'DISABLED'
      }`
    );

    console.log(
      `NIM API Key configured: ${
        NIM_API_KEY
          ? 'YES'
          : 'NO'
      }`
    );

    console.log(
      '========================================'
    );
  }
);

// Export for Vercel
module.exports = app;
```
