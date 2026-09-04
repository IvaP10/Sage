@preconcurrency import AVFoundation
import Foundation
@preconcurrency import Speech

@MainActor
final class VoiceInputController {
    enum Activation: Equatable {
        case microphoneButton
        case wakeWord
    }

    enum State: Equatable {
        case idle
        case wakeListening
        case listening(Activation)
        case processing
    }

    enum VoiceError: LocalizedError {
        case microphoneDenied
        case speechRecognitionDenied
        case onDeviceRecognitionUnavailable
        case audioInputUnavailable
        case recognizerUnavailable

        var errorDescription: String? {
            switch self {
            case .microphoneDenied:
                "Microphone access is off. You can enable it for Sage in System Settings → Privacy & Security → Microphone."
            case .speechRecognitionDenied:
                "Speech recognition access is off. You can enable it for Sage in System Settings → Privacy & Security → Speech Recognition."
            case .onDeviceRecognitionUnavailable:
                "On-device speech recognition is not available for the current language. Sage will not send background audio to a network service."
            case .audioInputUnavailable:
                "No usable microphone input is available."
            case .recognizerUnavailable:
                "On-device speech recognition is temporarily unavailable."
            }
        }
    }

    var onStateChange: ((State) -> Void)?
    var onTranscript: ((String) -> Void)?
    var onCommand: ((String, Activation) -> Void)?
    var onError: ((String) -> Void)?

    private(set) var state: State = .idle
    private(set) var wakeListeningEnabled = false
    private(set) var wakePhrase = "Hey Sage"

    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var inputTapInstalled = false
    private var commandTranscript = ""
    private var wakeTimeoutTask: Task<Void, Never>?
    private var silenceTask: Task<Void, Never>?
    private var restartTask: Task<Void, Never>?
    private var sessionGeneration = UUID()

    var microphoneAuthorization: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    var speechAuthorization: SFSpeechRecognizerAuthorizationStatus {
        SFSpeechRecognizer.authorizationStatus()
    }

    func configureWakeWord(enabled: Bool, phrase: String = "Hey Sage") {
        wakeListeningEnabled = enabled
        let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        wakePhrase = normalized.isEmpty ? "Hey Sage" : normalized

        if enabled {
            resumeWakeListeningIfAuthorized()
        } else if state == .wakeListening {
            stopRecognition(nextState: .idle)
        }
    }

    func resumeWakeListeningIfAuthorized() {
        guard wakeListeningEnabled,
              microphoneAuthorization == .authorized,
              speechAuthorization == .authorized,
              state == .idle else { return }
        do {
            try startRecognition(mode: .wakeListening)
        } catch {
            stopRecognition(nextState: .idle)
            onError?(error.localizedDescription)
        }
    }

    func startFromMicrophoneButton() async {
        do {
            try await requestPermissionsForImmediateUse()
            try startRecognition(mode: .listening(.microphoneButton))
        } catch {
            stopRecognition(nextState: .idle)
            onError?(error.localizedDescription)
        }
    }

    func finishCurrentCommand() {
        guard case .listening(let activation) = state else { return }
        completeCommand(activation: activation)
    }

    func cancelCurrentCommand() {
        stopRecognition(nextState: .idle)
        onTranscript?("")
        scheduleWakeRestart()
    }

    func stop() {
        wakeListeningEnabled = false
        stopRecognition(nextState: .idle)
    }

    private func requestPermissionsForImmediateUse() async throws {
        let microphoneGranted: Bool
        switch microphoneAuthorization {
        case .authorized:
            microphoneGranted = true
        case .notDetermined:
            microphoneGranted = await AVCaptureDevice.requestAccess(for: .audio)
        case .denied, .restricted:
            microphoneGranted = false
        @unknown default:
            microphoneGranted = false
        }
        guard microphoneGranted else { throw VoiceError.microphoneDenied }

        let speechStatus: SFSpeechRecognizerAuthorizationStatus
        switch speechAuthorization {
        case .authorized:
            speechStatus = .authorized
        case .notDetermined:
            speechStatus = await Self.requestSpeechAuthorization()
        case .denied, .restricted:
            speechStatus = speechAuthorization
        @unknown default:
            speechStatus = speechAuthorization
        }
        guard speechStatus == .authorized else { throw VoiceError.speechRecognitionDenied }
    }

    private func startRecognition(mode: State) throws {
        restartTask?.cancel()
        restartTask = nil
        stopRecognition(nextState: .idle)

        guard let recognizer = SFSpeechRecognizer(locale: .current), recognizer.isAvailable else {
            throw VoiceError.recognizerUnavailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw VoiceError.onDeviceRecognitionUnavailable
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw VoiceError.audioInputUnavailable
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        request.contextualStrings = [wakePhrase, "Sage"]

        let generation = UUID()
        sessionGeneration = generation
        commandTranscript = ""
        audioEngine = engine
        recognitionRequest = request

        Self.installAudioTap(on: input, format: format, request: request)
        inputTapInstalled = true

        recognitionTask = Self.startRecognitionTask(
            recognizer: recognizer,
            request: request
        ) { @MainActor [weak self] transcript, isFinal, errorMessage in
                self?.handleRecognition(
                    generation: generation,
                    transcript: transcript,
                    isFinal: isFinal,
                    errorMessage: errorMessage
                )
        }

        engine.prepare()
        try engine.start()
        setState(mode)
    }

    private func handleRecognition(
        generation: UUID,
        transcript: String?,
        isFinal: Bool,
        errorMessage: String?
    ) {
        guard generation == sessionGeneration else { return }

        if let transcript {
            switch state {
            case .wakeListening:
                if let command = commandAfterWakePhrase(in: transcript) {
                    commandTranscript = command
                    setState(.listening(.wakeWord))
                    onTranscript?(command)
                    scheduleWakeTimeout()
                    if !command.isEmpty { scheduleSilenceCompletion() }
                }
            case .listening(let activation):
                let visibleTranscript = activation == .wakeWord
                    ? commandAfterWakePhrase(in: transcript) ?? commandTranscript
                    : transcript
                commandTranscript = visibleTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
                onTranscript?(commandTranscript)
                if activation == .wakeWord, !commandTranscript.isEmpty {
                    scheduleSilenceCompletion()
                }
            case .idle, .processing:
                break
            }
        }

        if isFinal, case .listening(let activation) = state {
            completeCommand(activation: activation)
            return
        }

        if let errorMessage {
            let wasWakeListening = state == .wakeListening
            stopRecognition(nextState: .idle)
            if wasWakeListening {
                scheduleWakeRestart()
            } else {
                onError?(errorMessage)
            }
        }
    }

    private func commandAfterWakePhrase(in transcript: String) -> String? {
        let phrases = [wakePhrase, "Sage"]
        for phrase in phrases {
            let escaped = NSRegularExpression.escapedPattern(for: phrase)
            guard let range = transcript.range(
                of: "\\b\(escaped)\\b",
                options: [.caseInsensitive, .regularExpression]
            ) else { continue }
            return String(transcript[range.upperBound...])
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
        }
        return nil
    }

    private func completeCommand(activation: Activation) {
        let command = commandTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        stopRecognition(nextState: .processing)
        guard !command.isEmpty else {
            setState(.idle)
            onTranscript?("")
            scheduleWakeRestart()
            return
        }
        onCommand?(command, activation)
        scheduleWakeRestart()
    }

    private func scheduleWakeTimeout() {
        wakeTimeoutTask?.cancel()
        wakeTimeoutTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled else { return }
            self?.finishCurrentCommand()
        }
    }

    private func scheduleSilenceCompletion() {
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(1_650))
            guard !Task.isCancelled else { return }
            self?.finishCurrentCommand()
        }
    }

    private func scheduleWakeRestart() {
        restartTask?.cancel()
        restartTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(550))
            guard !Task.isCancelled else { return }
            self?.setState(.idle)
            self?.resumeWakeListeningIfAuthorized()
        }
    }

    private func stopRecognition(nextState: State) {
        wakeTimeoutTask?.cancel()
        wakeTimeoutTask = nil
        silenceTask?.cancel()
        silenceTask = nil

        if let engine = audioEngine {
            if inputTapInstalled {
                engine.inputNode.removeTap(onBus: 0)
            }
            engine.stop()
        }
        inputTapInstalled = false
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        audioEngine = nil
        sessionGeneration = UUID()
        setState(nextState)
    }

    private func setState(_ nextState: State) {
        guard state != nextState else { return }
        state = nextState
        onStateChange?(nextState)
    }

    /// TCC does not guarantee that its authorization callback runs on the main queue.
    /// Build the continuation from a nonisolated context so Swift does not attach a
    /// MainActor precondition to the callback that TCC invokes.
    private nonisolated static func requestSpeechAuthorization() async
        -> SFSpeechRecognizerAuthorizationStatus
    {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// AVAudioEngine invokes tap blocks on its realtime audio queue. The block must
    /// therefore be created outside MainActor isolation. Appending the captured
    /// buffer is the thread-safe handoff expected by SFSpeechAudioBufferRecognitionRequest.
    private nonisolated static func installAudioTap(
        on input: AVAudioInputNode,
        format: AVAudioFormat,
        request: SFSpeechAudioBufferRecognitionRequest
    ) {
        input.installTap(onBus: 0, bufferSize: 1_024, format: format) { buffer, _ in
            request.append(buffer)
        }
    }

    /// Speech recognition results may arrive on an arbitrary queue. Extract the
    /// immutable result there, then explicitly hop to MainActor before touching
    /// VoiceInputController state.
    private nonisolated static func startRecognitionTask(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechAudioBufferRecognitionRequest,
        deliver: @escaping @MainActor @Sendable (String?, Bool, String?) -> Void
    ) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, error in
            let transcript = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let errorMessage = error?.localizedDescription
            Task { @MainActor in
                deliver(transcript, isFinal, errorMessage)
            }
        }
    }
}
