@preconcurrency import AVFoundation
import Foundation
@preconcurrency import Speech

enum LocalWakeWordError: LocalizedError {
    case invalidPhrase
    case recognizerUnavailable
    case onDeviceRecognitionUnavailable

    var errorDescription: String? {
        switch self {
        case .invalidPhrase:
            "Use an English wake phrase with one to six words. Push-to-talk is still available."
        case .recognizerUnavailable:
            "On-device speech recognition is temporarily unavailable. Push-to-talk is still available."
        case .onDeviceRecognitionUnavailable:
            "On-device speech recognition is not available for this language. Sage will not send wake audio to a network service."
        }
    }
}

/// Immutable audio snapshot copied from AVAudioEngine's realtime callback.
/// The source buffer is only read after the callback has returned.
final class VoiceAudioBuffer: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer

    init?(copying source: AVAudioPCMBuffer) {
        guard let copy = AVAudioPCMBuffer(
            pcmFormat: source.format,
            frameCapacity: source.frameLength
        ) else { return nil }
        copy.frameLength = source.frameLength

        let sourceBuffers = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: source.audioBufferList)
        )
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceBuffers.count == destinationBuffers.count else { return nil }
        for index in sourceBuffers.indices {
            guard let sourceData = sourceBuffers[index].mData,
                  let destinationData = destinationBuffers[index].mData else { return nil }
            let byteCount = Int(sourceBuffers[index].mDataByteSize)
            memcpy(destinationData, sourceData, byteCount)
            destinationBuffers[index].mDataByteSize = sourceBuffers[index].mDataByteSize
        }
        buffer = copy
    }
}

/// Sage-owned phrase matching over text returned by Apple's on-device recognizer.
/// Matching token sequences avoids treating a phrase embedded inside a longer word
/// as a wake event (for example, "sage" inside "sagely").
enum WakePhraseMatcher {
    static func tokens(_ text: String) -> [String] {
        let words = text.split { character in
            !character.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
        }
        return words.map {
            String($0).folding(
                options: [.caseInsensitive, .diacriticInsensitive],
                locale: Locale(identifier: "en_US_POSIX")
            )
        }
    }

    static func isValidPhrase(_ phrase: String) -> Bool {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 60,
              trimmed.unicodeScalars.allSatisfy(\.isASCII) else { return false }
        let phraseTokens = tokens(trimmed)
        return (1...6).contains(phraseTokens.count)
            && phraseTokens.allSatisfy { !$0.isEmpty }
    }

    static func contains(_ phrase: String, in transcript: String) -> Bool {
        let needle = tokens(phrase)
        let haystack = tokens(transcript)
        guard !needle.isEmpty, needle.count <= haystack.count else { return false }
        return haystack.indices.contains { start in
            let end = start + needle.count
            return end <= haystack.count && Array(haystack[start..<end]) == needle
        }
    }
}

/// On-device wake listening uses the native Speech interface. Sage owns phrase
/// validation, exact token matching, lifecycle, restart policy, and interruption.
/// Raw audio stays in process memory and is never written to disk.
@MainActor
final class LocalWakeWordDetector {
    var onDetected: (@MainActor @Sendable () -> Void)?
    var onFailure: (@MainActor @Sendable (Error) -> Void)?

    let phrase: String

    private let recognizer: SFSpeechRecognizer
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var generation = UUID()
    private var isRunning = false
    private var didDetect = false
    private var consecutiveFailures = 0

    static func load(phrase: String) async throws -> LocalWakeWordDetector {
        try LocalWakeWordDetector(phrase: phrase)
    }

    private init(phrase: String) throws {
        let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard WakePhraseMatcher.isValidPhrase(normalized) else {
            throw LocalWakeWordError.invalidPhrase
        }
        guard let recognizer = SFSpeechRecognizer(locale: .current), recognizer.isAvailable else {
            throw LocalWakeWordError.recognizerUnavailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw LocalWakeWordError.onDeviceRecognitionUnavailable
        }
        self.phrase = normalized
        self.recognizer = recognizer
    }

    func process(_ audio: VoiceAudioBuffer) {
        guard isRunning, !didDetect else { return }
        request?.append(audio.buffer)
    }

    func pause() {
        isRunning = false
        generation = UUID()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
    }

    func resume() {
        guard !isRunning else { return }
        isRunning = true
        didDetect = false
        consecutiveFailures = 0
        beginRecognition()
    }

    private func beginRecognition() {
        guard isRunning, !didDetect else { return }
        guard recognizer.isAvailable else {
            fail(LocalWakeWordError.recognizerUnavailable)
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .search
        request.contextualStrings = [phrase, "Sage"]
        self.request = request

        let taskGeneration = UUID()
        generation = taskGeneration
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            let transcript = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let errorMessage = error?.localizedDescription
            Task { @MainActor [weak self] in
                self?.receive(
                    transcript: transcript,
                    isFinal: isFinal,
                    errorMessage: errorMessage,
                    generation: taskGeneration
                )
            }
        }
    }

    private func receive(
        transcript: String?,
        isFinal: Bool,
        errorMessage: String?,
        generation taskGeneration: UUID
    ) {
        guard isRunning, generation == taskGeneration, !didDetect else { return }
        if let transcript, WakePhraseMatcher.contains(phrase, in: transcript) {
            didDetect = true
            isRunning = false
            generation = UUID()
            request?.endAudio()
            task?.cancel()
            task = nil
            request = nil
            onDetected?()
            return
        }

        guard isFinal || errorMessage != nil else { return }
        task = nil
        request = nil
        let restartGeneration = UUID()
        generation = restartGeneration
        if errorMessage != nil {
            consecutiveFailures += 1
            if consecutiveFailures >= 2 {
                fail(LocalWakeWordError.recognizerUnavailable)
                return
            }
        } else {
            consecutiveFailures = 0
        }

        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            guard let self, self.isRunning, self.generation == restartGeneration else { return }
            self.beginRecognition()
        }
    }

    private func fail(_ error: Error) {
        pause()
        onFailure?(error)
    }
}
