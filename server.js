// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const GLM_REASONING_EFFORT = 'low';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

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

function getModelList() {
  const created = Math.floor(Date.now() / 1000);

  return Object.keys(MODEL_MAPPING).map((model) => ({
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
    Pragma: 'no-cache',
    Expires: '0'
  });

  res.json({
    object: 'list',
    data: models
  });
}

app.get('/v1/models', sendModelList);
app.get('/models', sendModelList);

app.options('/v1/models', (req, res) => res.sendStatus(204));
app.options('/models', (req, res) => res.sendStatus(204));

app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '1.2.1',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions',
      completions: '/v1/completions'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_configured: !!NIM_API_KEY
  });
});

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
        messages: [
          {
            role: 'user',
            content: 'test'
          }
        ],
        max_tokens: 1
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        validateStatus: (status) => status < 500
      }
    );

    if (
      testResponse.status >= 200 &&
      testResponse.status < 300
    ) {
      nimModel = model;
      console.log(
        `Model ${model} is directly supported by NIM`
      );
    }
  } catch (e) {
    console.log(
      'Model test failed, using fallback logic'
    );
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


// These determine which fields the proxy itself consumes.
//
// IMPORTANT:
// We intentionally DO NOT have a sampler whitelist.
// Everything else is forwarded to NVIDIA NIM exactly as supplied
// by SillyTavern.

const CHAT_EXCLUDED_FIELDS = new Set([
  'model',
  'messages'
]);

const TEXT_EXCLUDED_FIELDS = new Set([
  'model',
  'prompt',
  'messages'
]);

function forwardParameters(
  source,
  target,
  excludedFields
) {
  for (const [key, value] of Object.entries(source)) {
    if (excludedFields.has(key)) {
      continue;
    }

    target[key] = value;
  }

  return target;
}


function applyModelSpecificParameters(
  nimRequest,
  nimModel
) {
  // No sampler whitelist.
  //
  // Arbitrary parameters supplied by SillyTavern are passed
  // through to NIM. If NIM supports them, they can be used.
  // If NIM rejects them, the actual NIM error is returned.

  if (
    nimModel === 'z-ai/glm-5.3' ||
    nimModel === 'z-ai/glm-5.3-flash'
  ) {
    // Don't overwrite an explicitly supplied reasoning_effort.
    if (nimRequest.reasoning_effort === undefined) {
      nimRequest.reasoning_effort =
        GLM_REASONING_EFFORT;
    }
  }

  if (
    ENABLE_THINKING_MODE &&
    nimModel !== 'z-ai/glm-5.3' &&
    nimModel !== 'z-ai/glm-5.3-flash'
  ) {
    // Don't overwrite an existing extra_body.
    if (!nimRequest.extra_body) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true
        }
      };
    }
  }
}


function handleNimResponseError(
  error,
  res
) {
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;

    console.error(
      `[NIM ERROR] HTTP ${status}:`,
      JSON.stringify(data, null, 2)
    );

    return res.status(status).json(data);
  }

  console.error(
    '[PROXY ERROR]',
    error.message
  );

  return res.status(500).json({
    error: {
      message: error.message,
      type: 'proxy_error'
    }
  });
}


async function streamNimResponse(
  nimResponse,
  res,
  mode = 'chat'
) {
  res.setHeader(
    'Content-Type',
    'text/event-stream'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  res.flushHeaders();

  let buffer = '';

  nimResponse.data.on(
    'data',
    (chunk) => {
      buffer += chunk.toString();

      const lines = buffer.split('\n');

      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();

        if (!trimmed) {
          continue;
        }

        if (trimmed === 'data: [DONE]') {
          if (mode === 'text') {
            res.write(
              'data: [DONE]\n\n'
            );
          } else {
            res.write(
              'data: [DONE]\n\n'
            );
          }

          continue;
        }

        if (!trimmed.startsWith('data:')) {
          continue;
        }

        const jsonText =
          trimmed.slice(5).trim();

        if (!jsonText) {
          continue;
        }

        try {
          const data =
            JSON.parse(jsonText);

          if (mode === 'text') {
            // Convert an OpenAI chat-completion
            // streaming chunk into a Text Completion
            // streaming chunk for SillyTavern.

            const choice =
              data.choices &&
              data.choices[0];

            const delta =
              choice &&
              choice.delta;

            const content =
              delta &&
              delta.content;

            if (content) {
              const textChunk = {
                id:
                  data.id ||
                  `chatcmpl-${Date.now()}`,
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
                    text: content,
                    index:
                      choice.index || 0,
                    logprobs: null,
                    finish_reason:
                      choice.finish_reason ||
                      null
                  }
                ]
              };

              res.write(
                `data: ${JSON.stringify(
                  textChunk
                )}\n\n`
              );
            }
          } else {
            // Normal Chat Completion stream.

            res.write(
              `data: ${JSON.stringify(
                data
              )}\n\n`
            );
          }
        } catch (parseError) {
          console.log(
            '[STREAM] Could not parse chunk:',
            jsonText
          );
        }
      }
    }
  );

  nimResponse.data.on(
    'end',
    () => {
      res.end();
    }
  );

  nimResponse.data.on(
    'error',
    (error) => {
      console.error(
        '[STREAM ERROR]',
        error
      );

      if (!res.headersSent) {
        res.status(500);
      }

      res.end();
    }
  );
}


function createOpenAIChatResponse(
  nimData
) {
  return {
    id:
      nimData.id ||
      `chatcmpl-${Date.now()}`,

    object:
      'chat.completion',

    created:
      nimData.created ||
      Math.floor(
        Date.now() / 1000
      ),

    model:
      nimData.model || '',

    choices:
      nimData.choices || [],

    usage:
      nimData.usage
  };
}


// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    try {
      if (!NIM_API_KEY) {
        return res.status(500).json({
          error: {
            message:
              'NIM_API_KEY is not configured',
            type:
              'configuration_error'
          }
        });
      }

      const body = req.body || {};

      const requestedModel =
        body.model;

      if (!requestedModel) {
        return res.status(400).json({
          error: {
            message:
              'Missing model',
            type:
              'invalid_request_error'
          }
        });
      }

      const nimModel =
        await resolveModel(
          requestedModel
        );

      const nimRequest = {
        model: nimModel,
        messages:
          body.messages || []
      };

      // Forward EVERYTHING except fields controlled
      // by the proxy itself.
      forwardParameters(
        body,
        nimRequest,
        CHAT_EXCLUDED_FIELDS
      );

      applyModelSpecificParameters(
        nimRequest,
        nimModel
      );

      console.log(
        `[CHAT] ${requestedModel} -> ${nimModel}`
      );

      console.log(
        '[CHAT] Forwarded parameters:',
        Object.keys(nimRequest)
      );

      const isStreaming =
        body.stream === true;

      const nimResponse =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,
              'Content-Type':
                'application/json',
              Accept: isStreaming
                ? 'text/event-stream'
                : 'application/json'
            },

            responseType:
              isStreaming
                ? 'stream'
                : 'json',

            timeout: 0,

            validateStatus:
              (status) => status < 500
          }
        );

      if (
        nimResponse.status < 200 ||
        nimResponse.status >= 300
      ) {
        console.error(
          `[NIM CHAT ERROR] HTTP ${nimResponse.status}:`,
          JSON.stringify(
            nimResponse.data
          )
        );

        if (
          nimResponse.data &&
          typeof nimResponse.data.pipe ===
            'function'
        ) {
          let errorBody = '';

          nimResponse.data.on(
            'data',
            (chunk) => {
              errorBody +=
                chunk.toString();
            }
          );

          nimResponse.data.on(
            'end',
            () => {
              try {
                const parsed =
                  JSON.parse(errorBody);

                res
                  .status(nimResponse.status)
                  .json(parsed);
              } catch {
                res
                  .status(nimResponse.status)
                  .send(errorBody);
              }
            }
          );

          return;
        }

        return res
          .status(nimResponse.status)
          .json(nimResponse.data);
      }

      if (isStreaming) {
        return streamNimResponse(
          nimResponse,
          res,
          'chat'
        );
      }

      const response =
        createOpenAIChatResponse(
          nimResponse.data
        );

      return res.json(response);
    } catch (error) {
      return handleNimResponseError(
        error,
        res
      );
    }
  }
);


// ============================================================
// TEXT COMPLETIONS -> CHAT COMPLETIONS ADAPTER
// ============================================================
//
// SillyTavern's Text Completion endpoint sends:
//
// {
//   model: "...",
//   prompt: "...",
//   temperature: ...,
//   top_p: ...,
//   top_k: ...,
//   min_p: ...,
//   repetition_penalty: ...,
//   ...
// }
//
// NVIDIA NIM uses Chat Completions.
//
// We therefore convert:
//
// prompt
//   -> messages: [{ role: "user", content: prompt }]
//
// Every other parameter is passed through unchanged.
// ============================================================

app.post(
  '/v1/completions',
  async (req, res) => {
    try {
      if (!NIM_API_KEY) {
        return res.status(500).json({
          error: {
            message:
              'NIM_API_KEY is not configured',
            type:
              'configuration_error'
          }
        });
      }

      const body = req.body || {};

      const requestedModel =
        body.model;

      if (!requestedModel) {
        return res.status(400).json({
          error: {
            message:
              'Missing model',
            type:
              'invalid_request_error'
          }
        });
      }

      const prompt =
        body.prompt !== undefined
          ? body.prompt
          : '';

      const nimModel =
        await resolveModel(
          requestedModel
        );

      const nimRequest = {
        model: nimModel,

        messages: [
          {
            role: 'user',
            content: prompt
          }
        ]
      };

      // Forward every Text Completion parameter
      // except model/prompt/messages.
      //
      // This intentionally includes experimental
      // sampler parameters such as:
      //
      // repetition_penalty
      // min_p
      // top_k
      // top_p
      // temperature
      // frequency_penalty
      // presence_penalty
      // etc.
      //
      // NIM gets the parameters and decides whether
      // they are supported.

      forwardParameters(
        body,
        nimRequest,
        TEXT_EXCLUDED_FIELDS
      );

      applyModelSpecificParameters(
        nimRequest,
        nimModel
      );

      console.log(
        `[TEXT] ${requestedModel} -> ${nimModel}`
      );

      console.log(
        '[TEXT] Forwarded parameters:',
        Object.keys(nimRequest)
      );

      const isStreaming =
        body.stream === true;

      const nimResponse =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,
              'Content-Type':
                'application/json',
              Accept: isStreaming
                ? 'text/event-stream'
                : 'application/json'
            },

            responseType:
              isStreaming
                ? 'stream'
                : 'json',

            timeout: 0,

            validateStatus:
              (status) => status < 500
          }
        );

      if (
        nimResponse.status < 200 ||
        nimResponse.status >= 300
      ) {
        console.error(
          `[NIM TEXT ERROR] HTTP ${nimResponse.status}:`,
          JSON.stringify(
            nimResponse.data
          )
        );

        if (
          nimResponse.data &&
          typeof nimResponse.data.pipe ===
            'function'
        ) {
          let errorBody = '';

          nimResponse.data.on(
            'data',
            (chunk) => {
              errorBody +=
                chunk.toString();
            }
          );

          nimResponse.data.on(
            'end',
            () => {
              try {
                const parsed =
                  JSON.parse(errorBody);

                res
                  .status(nimResponse.status)
                  .json(parsed);
              } catch {
                res
                  .status(nimResponse.status)
                  .send(errorBody);
              }
            }
          );

          return;
        }

        return res
          .status(nimResponse.status)
          .json(nimResponse.data);
      }

      if (isStreaming) {
        return streamNimResponse(
          nimResponse,
          res,
          'text'
        );
      }

      const chatData =
        nimResponse.data;

      const choice =
        chatData.choices &&
        chatData.choices[0];

      const message =
        choice &&
        choice.message;

      const generatedText =
        message &&
        message.content
          ? message.content
          : '';

      const completionResponse = {
        id:
          chatData.id ||
          `cmpl-${Date.now()}`,

        object:
          'text_completion',

        created:
          chatData.created ||
          Math.floor(
            Date.now() / 1000
          ),

        model:
          requestedModel,

        choices: [
          {
            text: generatedText,
            index: 0,
            logprobs: null,
            finish_reason:
              choice &&
              choice.finish_reason
                ? choice.finish_reason
                : null
          }
        ],

        usage:
          chatData.usage
      };

      return res.json(
        completionResponse
      );
    } catch (error) {
      return handleNimResponseError(
        error,
        res
      );
    }
  }
);


// ============================================================
// CATCH-ALL
// ============================================================
//
// Using app.use() instead of app.all('*', ...) avoids wildcard
// route parsing problems with newer Express versions.
// ============================================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message:
        `Endpoint not found: ${req.method} ${req.originalUrl}`,
      type:
        'not_found'
    }
  });
});


app.listen(
  PORT,
  () => {
    console.log(
      `NVIDIA NIM Proxy listening on port ${PORT}`
    );

    console.log(
      `NVIDIA API base: ${NIM_API_BASE}`
    );

    console.log(
      `NIM API key configured: ${!!NIM_API_KEY}`
    );

    console.log(
      `Chat endpoint: /v1/chat/completions`
    );

    console.log(
      `Text endpoint: /v1/completions`
    );

    console.log(
      `Model endpoint: /v1/models`
    );

    console.log(
      `Transparent parameter forwarding: ENABLED`
    );
  }
);
