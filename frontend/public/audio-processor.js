// AudioWorklet processor for capturing microphone audio and sending bounded chunks
// This replaces the deprecated ScriptProcessorNode

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    
    // Configuration
    this.chunkSize = 1600; // 100ms at 16kHz
    this.sampleRate = 16000;
    this.buffer = new Float32Array(this.chunkSize);
    this.bufferIndex = 0;
    this.sequenceNumber = 0;
    
    // Handle messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'configure') {
        if (event.data.chunkSize) this.chunkSize = event.data.chunkSize;
        if (event.data.sampleRate) this.sampleRate = event.data.sampleRate;
        this.buffer = new Float32Array(this.chunkSize);
        this.bufferIndex = 0;
      } else if (event.data.type === 'flush') {
        this.flushBuffer();
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    
    if (input.length > 0) {
      const channelData = input[0];
      
      for (let i = 0; i < channelData.length; i++) {
        this.buffer[this.bufferIndex] = channelData[i];
        this.bufferIndex++;
        
        if (this.bufferIndex >= this.chunkSize) {
          this.flushBuffer();
        }
      }
    }
    
    // Keep the processor alive
    return true;
  }

  flushBuffer() {
    if (this.bufferIndex > 0) {
      // Send only the filled portion
      const chunk = this.buffer.slice(0, this.bufferIndex);
      this.port.postMessage({
        type: 'audio_chunk',
        sequence: this.sequenceNumber++,
        sample_rate: this.sampleRate,
        channels: 1,
        pcm_f32: chunk
      });
      this.bufferIndex = 0;
    }
  }
}

registerProcessor('audio-processor', AudioProcessor);