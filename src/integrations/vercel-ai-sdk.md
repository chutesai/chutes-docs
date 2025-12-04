# Vercel AI SDK Integration

The **Chutes.ai Provider for Vercel AI SDK** allows you to use open-source AI models hosted on Chutes.ai with the Vercel AI SDK. It supports a wide range of capabilities including chat, streaming, tool calling, and multimodal generation.

## Features

- ✅ **Language Models**: Complete support for chat and text completion
- ✅ **Streaming**: Real-time Server-Sent Events (SSE) streaming
- ✅ **Tool Calling**: Full function/tool calling support
- ✅ **Multimodal**: Image, Video, Audio (TTS/STT/Music) generation
- ✅ **Chute Warmup**: Pre-warm chutes for instant response times
- ✅ **Type-Safe**: Fully typed for excellent IDE support

## Installation

Install the provider and the AI SDK:

```bash
npm install @chutes-ai/ai-sdk-provider ai
```

**Note**: For Next.js projects with TypeScript, AI SDK v5 is recommended:

```bash
npm install @chutes-ai/ai-sdk-provider ai@^5.0.0
```

## Configuration

### 1. Get API Key

Get your API key from [Chutes.ai](https://chutes.ai) and set it as an environment variable:

```bash
export CHUTES_API_KEY=your-api-key-here
```

### 2. Initialize Provider

You can initialize the provider with your API key.

```typescript
import { createChutes } from '@chutes-ai/ai-sdk-provider';

const chutes = createChutes({
  apiKey: process.env.CHUTES_API_KEY,
});
```

## Language Models

### Text Generation

Generate text using any LLM hosted on Chutes.

```typescript
import { generateText } from 'ai';

const model = chutes('https://chutes-deepseek-ai-deepseek-v3.chutes.ai');

const result = await generateText({
  model,
  prompt: 'Explain quantum computing in simple terms',
});

console.log(result.text);
```

### Streaming Responses

Stream responses in real-time for a better user experience.

```typescript
import { streamText } from 'ai';

const result = await streamText({
  model: chutes('https://chutes-meta-llama-llama-3-1-70b-instruct.chutes.ai'),
  prompt: 'Write a story about a space traveler.',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### Tool Calling

Connect LLMs to external data and functions.

```typescript
import { z } from 'zod';

const result = await generateText({
  model: chutes('https://chutes-deepseek-ai-deepseek-v3.chutes.ai'),
  tools: {
    getWeather: {
      description: 'Get the current weather',
      parameters: z.object({
        location: z.string().describe('City name'),
      }),
      execute: async ({ location }) => {
        return { temp: 72, condition: 'Sunny', location };
      },
    },
  },
  prompt: 'What is the weather in San Francisco?',
});
```

## Multimodal Capabilities

### Image Generation

Generate images using models like FLUX.

```typescript
import * as fs from 'fs';

const imageModel = chutes.imageModel('flux-dev');

const result = await imageModel.doGenerate({
  prompt: 'A cyberpunk city with neon lights and flying cars',
  size: '1024x1024',
});

const base64Data = result.images[0].split(',')[1];
fs.writeFileSync('city.png', Buffer.from(base64Data, 'base64'));
```

### Text-to-Speech (TTS)

Convert text to speech using over 50 available voices.

```typescript
const audioModel = chutes.audioModel('your-tts-chute-id');

const result = await audioModel.textToSpeech({
  text: 'Welcome to the future of AI.',
  voice: 'af_bella', // American Female - Bella
});

fs.writeFileSync('output.mp3', result.audio);
```

### Speech-to-Text (STT)

Transcribe audio files.

```typescript
const audioModel = chutes.audioModel('your-stt-chute-id');
const audioBuffer = fs.readFileSync('recording.mp3');

const transcription = await audioModel.speechToText({
  audio: audioBuffer,
  language: 'en',
});

console.log(transcription.text);
```

## Advanced Features

### Chute Warmup (Therm)

Pre-warm chutes to eliminate cold starts.

```typescript
// Warm up a chute
const result = await chutes.therm.warmup('your-chute-id');

if (result.isHot) {
  console.log('Chute is ready!');
} else {
  console.log('Warming up...');
}
```

### Embeddings

Generate vector embeddings for semantic search.

```typescript
import { embedMany } from 'ai';

const embeddingModel = chutes.textEmbeddingModel('text-embedding-3-small');

const { embeddings } = await embedMany({
  model: embeddingModel,
  values: ['Hello world', 'Machine learning is cool'],
});
```

## Troubleshooting

### Common Issues

- **404 Not Found**: Verify the chute URL is correct and the chute is deployed.
- **401 Unauthorized**: Check your `CHUTES_API_KEY`.
- **429 Rate Limit**: Implement exponential backoff or request a quota increase.

### Getting Help

- Check the [GitHub Repository](https://github.com/chutesai/ai-sdk-provider-chutes) for issues.
- Join the [Discord Community](https://discord.gg/chutes).

