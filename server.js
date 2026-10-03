// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const GLM_REASONING_EFFORT = 'low';

// Automatic recovery for transient/empty NIM responses.
const MAX_NIM_RETRIES = 2; // 2 retries = up to 3 total attempts
const RETRY_DELAYS_MS = [750, 1500];
const FIRST_TOKEN_TIMEOUT_MS = 45000;

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

const RETRYABLE_STATUS_CODES = new Set([
  408,
  429,
  500,
  502,
  503,
  504
]);

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
    version: '1.3.0',
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
    nim_api_configured: !!NIM_API_KEY,
    automatic_retries: MAX_NIM_RETRIES,
    first_token_timeout_ms: FIRST_TOKEN_TIMEOUT_MS
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


// ============================================================
// RETRY / RECOVERY HELPERS
// ============================================================

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return RETRYABLE_STATUS_CODES.has(status);
}

function isRetryableNetworkError(error) {
  if (error.response) {
    return false;
  }

  const code = error.code;

  return [
    'ECONNRESET',
    'ECONNABORTED',
    'ETIMEDOUT',
    'EPIPE',
    'ENETUNREACH',
    'EAI_AGAIN'
  ].includes(code);
}

function isUsableChatResponse(data) {
  const choice = data?.choices?.[0];

  if (!choice) {
    return false;
  }

  const content = choice.message?.content;

  if (
    typeof content === 'string' &&
    content.trim().length > 0
  ) {
    return true;
  }

  // A tool call can be a legitimate non-text completion.
  if (
    Array.isArray(choice.message?.tool_calls) &&
    choice.message.tool_calls.length > 0
  ) {
    return true;
  }

  return false;
}

function getTextCompletionText(data) {
  const choice = data?.choices?.[0];
  const content = choice?.message?.content;

  return typeof content === 'string'
    ? content
    : '';
}

async function readNimErrorStream(stream) {
  return new Promise((resolve) => {
    let body = '';

    stream.on('data', (chunk) => {
      body += chunk.toString();
    });

    stream.on('end', () => resolve(body));

    stream.on('error', () => resolve(body));
  });
}

function formatNimErrorBody(body) {
  try {
    return JSON.parse(body);
  } catch {
    return body;
  }
}

function logRetry(attempt, reason) {
  const nextAttempt = attempt + 1;

  const delay =
    RETRY_DELAYS_MS[attempt - 1] ||
    RETRY_DELAYS_MS[
      RETRY_DELAYS_MS.length - 1
    ];

  console.warn(
    `[RETRY] ${reason} | retry ${nextAttempt}/${MAX_NIM_RETRIES} in ${delay}ms`
  );

  return delay;
}


// ============================================================
// NON-STREAMING NIM REQUEST WITH RETRIES
// ============================================================

async function requestNimWithRetries(
  nimRequest,
  isStreaming,
  mode
) {
  for (
    let attempt = 1;
    attempt <= MAX_NIM_RETRIES + 1;
    attempt++
  ) {
    try {
      const response = await axios.post(
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

          validateStatus: () => true
        }
      );

      // Successful HTTP response.
      //
      // We still check the generated content because NIM can
      // occasionally return HTTP 200 without actually producing
      // useful text.

      if (
        response.status >= 200 &&
        response.status < 300
      ) {
        const usable = isStreaming
          ? true
          : mode === 'text'
            ? getTextCompletionText(
                response.data
              ).trim().length > 0
            : isUsableChatResponse(
                response.data
              );

        if (usable) {
          return response;
        }

        console.warn(
          `[NIM ${mode.toUpperCase()}] HTTP 200 but no usable generated content was returned.`
        );

        if (attempt <= MAX_NIM_RETRIES) {
          const delay = logRetry(
            attempt,
            'NIM returned an empty/invalid successful completion'
          );

          await sleep(delay);
          continue;
        }

        return {
          failed: true,
          status: 502,
          data: {
            error: {
              message:
                'NVIDIA NIM returned an empty or invalid completion after automatic retries.',
              type:
                'empty_generation'
            }
          }
        };
      }

      let errorData = response.data;

      if (
        errorData &&
        typeof errorData.pipe ===
          'function'
      ) {
        const errorBody =
          await readNimErrorStream(
            errorData
          );

        errorData =
          formatNimErrorBody(
            errorBody
          );
      }

      console.error(
        `[NIM ${mode.toUpperCase()} ERROR] HTTP ${response.status}:`,
        typeof errorData === 'string'
          ? errorData
          : JSON.stringify(
              errorData,
              null,
              2
            )
      );

      if (
        attempt <= MAX_NIM_RETRIES &&
        isRetryableStatus(
          response.status
        )
      ) {
        const delay = logRetry(
          attempt,
          `NIM HTTP ${response.status}`
        );

        await sleep(delay);
        continue;
      }

      return {
        failed: true,
        status: response.status,
        data: errorData
      };
    } catch (error) {
      const retryable =
        isRetryableNetworkError(
          error
        );

      console.error(
        `[NIM ${mode.toUpperCase()} REQUEST ERROR]`,
        error.code ||
          error.message
      );

      if (
        attempt <= MAX_NIM_RETRIES &&
        retryable
      ) {
        const delay = logRetry(
          attempt,
          `NIM network error ${
            error.code ||
            error.message
          }`
        );

        await sleep(delay);
        continue;
      }

      throw error;
    }
  }

  throw new Error(
    'NIM retry loop ended unexpectedly'
  );
}


function sendNimFailure(
  res,
  response,
  label
) {
  console.error(
    `[NIM ${label.toUpperCase()} FINAL ERROR] HTTP ${response.status}:`,
    typeof response.data === 'string'
      ? response.data
      : JSON.stringify(
          response.data,
          null,
          2
        )
  );

  return res
    .status(response.status)
    .json(response.data);
}


// ============================================================
// STREAMING
// ============================================================

function setStreamingHeaders(res) {
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
}


function parseSSELine(
  line,
  mode,
  onContent,
  onDone,
  onRawData
) {
  const trimmed = line.trim();

  if (!trimmed) {
    return;
  }

  if (
    trimmed ===
    'data: [DONE]'
  ) {
    onDone();
    return;
  }

  if (
    !trimmed.startsWith('data:')
  ) {
    return;
  }

  const jsonText =
    trimmed
      .slice(5)
      .trim();

  if (!jsonText) {
    return;
  }

  try {
    const data =
      JSON.parse(
        jsonText
      );

    const choice =
      data.choices?.[0];

    const delta =
      choice?.delta;

    const content =
      delta?.content;

    if (content) {
      if (mode === 'text') {
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
              text:
                content,

              index:
                choice.index ||
                0,

              logprobs:
                null,

              finish_reason:
                choice.finish_reason ||
                null
            }
          ]
        };

        onContent(
          `data: ${JSON.stringify(
            textChunk
          )}\n\n`
        );
      } else {
        onContent(
          `data: ${JSON.stringify(
            data
          )}\n\n`
        );
      }
    } else if (
      mode === 'chat'
    ) {
      // Preserve non-content chat chunks
      // such as role/finish metadata.
      onRawData(
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


function consumeStreamAttempt(
  nimResponse,
  res,
  mode
) {
  return new Promise(
    (resolve) => {
      const stream =
        nimResponse.data;

      let buffer = '';
      let outputStarted =
        false;

      let completed =
        false;

      let timedOut =
        false;

      const pendingOutput =
        [];

      let firstTokenTimer;


      const finish = (
        success,
        reason
      ) => {
        if (completed) {
          return;
        }

        completed = true;

        clearTimeout(
          firstTokenTimer
        );

        resolve({
          success,
          reason,
          outputStarted
        });
      };


      const flushPending =
        () => {
          if (
            !outputStarted ||
            !res.headersSent
          ) {
            return;
          }

          while (
            pendingOutput.length >
            0
          ) {
            res.write(
              pendingOutput.shift()
            );
          }
        };


      const emit =
        (data) => {
          if (!data) {
            return;
          }

          if (!outputStarted) {
            outputStarted =
              true;

            setStreamingHeaders(
              res
            );
          }

          pendingOutput.push(
            data
          );

          flushPending();
        };


      const emitRaw =
        (data) => {
          if (!data) {
            return;
          }

          // For Chat Completion mode, preserve
          // metadata chunks before/around content.
          //
          // Text Completion mode does not need these.

          if (
            mode === 'chat'
          ) {
            if (
              outputStarted
            ) {
              res.write(
                data
              );
            } else {
              pendingOutput.push(
                data
              );
            }
          }
        };


      const handleLine =
        (line) => {
          parseSSELine(
            line,
            mode,
            emit,
            () => {
              if (
                outputStarted
              ) {
                res.write(
                  'data: [DONE]\n\n'
                );

                finish(
                  true,
                  'done'
                );
              } else {
                finish(
                  false,
                  'NIM returned [DONE] without generated content'
                );
              }
            },
            emitRaw
          );
        };


      firstTokenTimer =
        setTimeout(
          () => {
            timedOut =
              true;

            console.warn(
              `[RETRY] NIM stream produced no usable text within ${FIRST_TOKEN_TIMEOUT_MS}ms`
            );

            stream.destroy();

            finish(
              false,
              'first-token timeout'
            );
          },
          FIRST_TOKEN_TIMEOUT_MS
        );


      stream.on(
        'data',
        (chunk) => {
          if (completed) {
            return;
          }

          buffer +=
            chunk.toString();

          const lines =
            buffer.split(
              '\n'
            );

          buffer =
            lines.pop() ||
            '';

          for (
            const line
            of lines
          ) {
            if (
              completed
            ) {
              break;
            }

            handleLine(
              line
            );
          }
        }
      );


      stream.on(
        'end',
        () => {
          if (completed) {
            return;
          }

          if (
            buffer.trim()
          ) {
            handleLine(
              buffer
            );
          }

          if (!completed) {
            if (
              outputStarted
            ) {
              finish(
                true,
                'stream ended'
              );
            } else {
              finish(
                false,
                'NIM stream ended without generated content'
              );
            }
          }
        }
      );


      stream.on(
        'error',
        (error) => {
          if (completed) {
            return;
          }

          if (timedOut) {
            finish(
              false,
              'first-token timeout'
            );

            return;
          }

          if (
            outputStarted
          ) {
            console.error(
              '[STREAM ERROR AFTER OUTPUT]',
              error.message
            );

            finish(
              true,
              'stream error after output'
            );
          } else {
            console.error(
              '[STREAM ERROR BEFORE OUTPUT]',
              error.message
            );

            finish(
              false,
              `stream error: ${
                error.code ||
                error.message
              }`
            );
          }
        }
      );
    }
  );
}


async function streamNimWithRetries(
  nimRequest,
  res,
  mode
) {
  for (
    let attempt = 1;
    attempt <= MAX_NIM_RETRIES + 1;
    attempt++
  ) {
    let nimResponse;

    try {
      nimResponse =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              'Content-Type':
                'application/json',

              Accept:
                'text/event-stream'
            },

            responseType:
              'stream',

            timeout:
              0,

            validateStatus:
              () => true
          }
        );
    } catch (error) {
      const retryable =
        isRetryableNetworkError(
          error
        );

      console.error(
        `[NIM ${mode.toUpperCase()} STREAM REQUEST ERROR]`,
        error.code ||
          error.message
      );

      if (
        attempt <=
          MAX_NIM_RETRIES &&
        retryable
      ) {
        const delay =
          logRetry(
            attempt,
            `NIM streaming network error ${
              error.code ||
              error.message
            }`
          );

        await sleep(
          delay
        );

        continue;
      }

      throw error;
    }


    if (
      nimResponse.status <
        200 ||
      nimResponse.status >=
        300
    ) {
      let errorData =
        nimResponse.data;

      if (
        errorData &&
        typeof errorData.pipe ===
          'function'
      ) {
        const errorBody =
          await readNimErrorStream(
            errorData
          );

        errorData =
          formatNimErrorBody(
            errorBody
          );
      }

      console.error(
        `[NIM ${mode.toUpperCase()} STREAM ERROR] HTTP ${nimResponse.status}:`,
        typeof errorData ===
          'string'
          ? errorData
          : JSON.stringify(
              errorData,
              null,
              2
            )
      );

      if (
        attempt <=
          MAX_NIM_RETRIES &&
        isRetryableStatus(
          nimResponse.status
        )
      ) {
        const delay =
          logRetry(
            attempt,
            `NIM streaming HTTP ${nimResponse.status}`
          );

        await sleep(
          delay
        );

        continue;
      }

      return res
        .status(
          nimResponse.status
        )
        .json(
          errorData
        );
    }


    const result =
      await consumeStreamAttempt(
        nimResponse,
        res,
        mode
      );


    if (
      result.success
    ) {
      if (
        !res.writableEnded
      ) {
        res.end();
      }

      return;
    }


    // Once actual text has reached ST, we cannot
    // safely retry because ST already has part of
    // the previous answer.

    if (
      result.outputStarted ||
      res.headersSent
    ) {
      console.warn(
        `[STREAM] Output had already started; ending instead of retrying: ${result.reason}`
      );

      if (
        !res.writableEnded
      ) {
        res.end();
      }

      return;
    }


    if (
      attempt <=
      MAX_NIM_RETRIES
    ) {
      const delay =
        logRetry(
          attempt,
          result.reason
        );

      await sleep(
        delay
      );

      continue;
    }


    // All retries failed before producing text.
    // Give ST a real error rather than an empty
    // successful response.

    console.error(
      `[STREAM] All ${
        MAX_NIM_RETRIES + 1
      } attempts failed before generated text.`
    );

    return res
      .status(502)
      .json({
        error: {
          message:
            'NVIDIA NIM failed to produce a usable response after automatic retries.',

          type:
            'upstream_generation_error',

          attempts:
            MAX_NIM_RETRIES + 1,

          reason:
            result.reason
        }
      });
  }
}


// ============================================================
// GENERIC ERROR HANDLER
// ============================================================

function handleNimResponseError(
  error,
  res
) {
  if (error.response) {
    const status =
      error.response.status;

    const data =
      error.response.data;

    console.error(
      `[NIM ERROR] HTTP ${status}:`,
      JSON.stringify(
        data,
        null,
        2
      )
    );

    return res
      .status(status)
      .json(data);
  }

  console.error(
    '[PROXY ERROR]',
    error.message
  );

  return res
    .status(500)
    .json({
      error: {
        message:
          error.message,

        type:
          'proxy_error'
      }
    });
}


// ============================================================
// OPENAI CHAT RESPONSE FORMAT
// ============================================================

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
        return res
          .status(500)
          .json({
            error: {
              message:
                'NIM_API_KEY is not configured',

              type:
                'configuration_error'
            }
          });
      }

      const body =
        req.body || {};

      const requestedModel =
        body.model;

      if (!requestedModel) {
        return res
          .status(400)
          .json({
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
        model:
          nimModel,

        messages:
          body.messages || []
      };

      // Forward EVERYTHING except fields
      // controlled by the proxy itself.

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
        Object.keys(
          nimRequest
        )
      );

      const isStreaming =
        body.stream === true;


      if (isStreaming) {
        return streamNimWithRetries(
          nimRequest,
          res,
          'chat'
        );
      }


      const nimResponse =
        await requestNimWithRetries(
          nimRequest,
          false,
          'chat'
        );

      if (
        nimResponse.failed
      ) {
        return sendNimFailure(
          res,
          nimResponse,
          'chat'
        );
      }


      const response =
        createOpenAIChatResponse(
          nimResponse.data
        );

      return res.json(
        response
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
        return res
          .status(500)
          .json({
            error: {
              message:
                'NIM_API_KEY is not configured',

              type:
                'configuration_error'
            }
          });
      }

      const body =
        req.body || {};

      const requestedModel =
        body.model;

      if (!requestedModel) {
        return res
          .status(400)
          .json({
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
        model:
          nimModel,

        messages: [
          {
            role:
              'user',

            content:
              prompt
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
        Object.keys(
          nimRequest
        )
      );

      const isStreaming =
        body.stream === true;


      if (isStreaming) {
        return streamNimWithRetries(
          nimRequest,
          res,
          'text'
        );
      }


      const nimResponse =
        await requestNimWithRetries(
          nimRequest,
          false,
          'text'
        );

      if (
        nimResponse.failed
      ) {
        return sendNimFailure(
          res,
          nimResponse,
          'text'
        );
      }


      const chatData =
        nimResponse.data;

      const generatedText =
        getTextCompletionText(
          chatData
        );


      const choice =
        chatData.choices &&
        chatData.choices[0];


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
            text:
              generatedText,

            index:
              0,

            logprobs:
              null,

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

app.use(
  (req, res) => {
    res
      .status(404)
      .json({
        error: {
          message:
            `Endpoint not found: ${req.method} ${req.originalUrl}`,

          type:
            'not_found'
        }
      });
  }
);


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

    console.log(
      `Automatic retries: ${MAX_NIM_RETRIES} retries (${MAX_NIM_RETRIES + 1} total attempts)`
    );

    console.log(
      `First-token timeout: ${FIRST_TOKEN_TIMEOUT_MS}ms`
    );
  }
);
