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
    var onWakeWordUnavailable: (() -> Void)?

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
    private var wakeStartupTask: Task<Void, Never>?
    private var wakeStartupGeneration = UUID()
    private var wakeListeningPaused = false
    private var sessionGeneration = UUID()
    private var wakeDetector: LocalWakeWordDetector?
    private var preRollBuffers: [VoiceAudioBuffer] = []
    private var preRollFrameCount = 0
    private let preRollDurationSeconds: Double = 2.2

    var microphoneAuthorization: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    var speechAuthorization: SFSpeechRecognizerAuthorizationStatus {
        SFSpeechRecognizer.authorizationStatus()
    }

    /// Called at launch with the saved preference. This never asks macOS for
    /// permission; only the settings toggle and push-to-talk button do that.
    func configureWakeWord(enabled: Bool, phrase: String = "Hey Sage") {
        let nextPhrase = Self.normalizedPhrase(phrase)
        let phraseChanged = nextPhrase != wakePhrase
        wakePhrase = nextPhrase
        wakeListeningEnabled = enabled
        if enabled { wakeListeningPaused = false }

        guard enabled else {
            wakeStartupGeneration = UUID()
            wakeStartupTask?.cancel()
            wakeStartupTask = nil
            wakeDetector?.pause()
            if state == .wakeListening { stopCapture(nextState: .idle) }
            return
        }

        if phraseChanged {
            wakeDetector?.pause()
            wakeDetector = nil
            if state == .wakeListening { stopCapture(nextState: .idle) }
        }
        resumeWakeListeningIfAuthorized()
    }

    /// Called only by the explicit wake-word settings action.
    func enableWakeWordFromUserAction(phrase: String) async {
        wakePhrase = Self.normalizedPhrase(phrase)
        wakeListeningEnabled = true
        wakeStartupGeneration = UUID()
        let generation = wakeStartupGeneration
        wakeStartupTask?.cancel()
        wakeStartupTask = nil
        do {
            try await requestPermissionsForImmediateUse()
            guard generation == wakeStartupGeneration, wakeListeningEnabled else { return }
            _ = try makeOnDeviceRecognizer()
            guard generation == wakeStartupGeneration, wakeListeningEnabled else { return }
            try await startWakeListening(generation: generation)
        } catch {
            guard generation == wakeStartupGeneration else { return }
            failWakeWord(error)
        }
    }

    func resumeWakeListeningIfAuthorized() {
        guard wakeListeningEnabled, !wakeListeningPaused, state == .idle else { return }
        guard microphoneAuthorization == .authorized,
              speechAuthorization == .authorized else {
            failWakeWord(
                microphoneAuthorization != .authorized
                    ? VoiceError.microphoneDenied
                    : VoiceError.speechRecognitionDenied
            )
            return
        }

        wakeStartupGeneration = UUID()
        let generation = wakeStartupGeneration
        wakeStartupTask?.cancel()
        wakeStartupTask = Task { [weak self] in
            guard let self else { return }
            do {
                _ = try self.makeOnDeviceRecognizer()
                try await self.startWakeListening(generation: generation)
            } catch {
                guard self.wakeStartupGeneration == generation else { return }
                self.failWakeWord(error)
            }
        }
    }

    func updateWakePhrase(_ phrase: String) {
        let nextPhrase = Self.normalizedPhrase(phrase)
        guard nextPhrase != wakePhrase else { return }
        wakePhrase = nextPhrase
        wakeDetector?.pause()
        wakeDetector = nil
        if state == .wakeListening { stopCapture(nextState: .idle) }
        if wakeListeningEnabled { resumeWakeListeningIfAuthorized() }
    }

    func startFromMicrophoneButton() async {
        do {
            try await requestPermissionsForImmediateUse()
            let recognizer = try makeOnDeviceRecognizer()
            wakeListeningPaused = false
            wakeStartupGeneration = UUID()
            wakeStartupTask?.cancel()
            wakeDetector?.pause()
            clearPreRoll()
            if audioEngine == nil { try startCapture() }
            startCommandRecognition(
                recognizer: recognizer,
                activation: .microphoneButton,
                preRoll: []
            )
        } catch {
            stopCapture(nextState: .idle)
            onError?(error.localizedDescription)
            if wakeListeningEnabled { scheduleWakeRestart() }
        }
    }

    func finishCurrentCommand() {
        guard case .listening(let activation) = state else { return }
        completeCommand(activation: activation)
    }

    func cancelCurrentCommand() {
        stopCapture(nextState: .idle)
        onTranscript?("")
        scheduleWakeRestart()
    }

    func pauseWakeListening() {
        wakeListeningPaused = true
        wakeStartupGeneration = UUID()
        wakeStartupTask?.cancel()
        wakeStartupTask = nil
        wakeDetector?.pause()
        guard state == .wakeListening else { return }
        stopCapture(nextState: .idle)
    }

    func resumeWakeListeningAfterSpeech() {
        wakeListeningPaused = false
        resumeWakeListeningIfAuthorized()
    }

    func stop() {
        wakeListeningEnabled = false
        wakeListeningPaused = false
        wakeStartupGeneration = UUID()
        wakeStartupTask?.cancel()
        wakeStartupTask = nil
        wakeDetector?.pause()
        stopCapture(nextState: .idle)
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

    private func startWakeListening(generation: UUID? = nil) async throws {
        guard wakeListeningEnabled, !wakeListeningPaused, state == .idle else { return }
        if let generation, wakeStartupGeneration != generation { return }
        let detector: LocalWakeWordDetector
        if let existing = wakeDetector, existing.phrase == wakePhrase {
            detector = existing
        } else {
            detector = try await LocalWakeWordDetector.load(phrase: wakePhrase)
            if let generation, wakeStartupGeneration != generation { return }
            guard wakeListeningEnabled, !wakeListeningPaused, state == .idle else { return }
            wakeDetector = detector
        }
        guard wakeListeningEnabled, !wakeListeningPaused, state == .idle else { return }
        detector.onDetected = { [weak self] in self?.handleWakeWordDetected() }
        detector.onFailure = { [weak self] error in self?.failWakeWord(error) }
        detector.resume()
        clearPreRoll()
        try startCapture()
        guard wakeListeningEnabled, state == .idle else {
            stopCapture(nextState: .idle)
            return
        }
        setState(.wakeListening)
    }

    private func makeOnDeviceRecognizer() throws -> SFSpeechRecognizer {
        guard let recognizer = SFSpeechRecognizer(locale: .current), recognizer.isAvailable else {
            throw VoiceError.recognizerUnavailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw VoiceError.onDeviceRecognitionUnavailable
        }
        return recognizer
    }

    private func startCapture() throws {
        guard audioEngine == nil else { return }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw VoiceError.audioInputUnavailable
        }
        Self.installAudioTap(on: input, format: format) { [weak self] audio in
            self?.handleCapturedAudio(audio)
        }
        inputTapInstalled = true
        audioEngine = engine
        engine.prepare()
        do {
            try engine.start()
        } catch {
            stopCapture(nextState: .idle)
            throw error
        }
    }

    private func handleCapturedAudio(_ audio: VoiceAudioBuffer) {
        switch state {
        case .wakeListening:
            appendToPreRoll(audio)
            wakeDetector?.process(audio)
        case .listening:
            recognitionRequest?.append(audio.buffer)
        case .idle, .processing:
            break
        }
    }

    private func handleWakeWordDetected() {
        guard state == .wakeListening, wakeListeningEnabled else { return }
        let recognizer: SFSpeechRecognizer
        do { recognizer = try makeOnDeviceRecognizer() }
        catch { failWakeWord(error); return }
        wakeDetector?.pause()
        let preRoll = preRollBuffers
        clearPreRoll()
        startCommandRecognition(
            recognizer: recognizer,
            activation: .wakeWord,
            preRoll: preRoll
        )
        scheduleWakeTimeout()
    }

    private func startCommandRecognition(
        recognizer: SFSpeechRecognizer,
        activation: Activation,
        preRoll: [VoiceAudioBuffer]
    ) {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        request.contextualStrings = [wakePhrase, "Sage"]

        let generation = UUID()
        sessionGeneration = generation
        commandTranscript = ""
        recognitionRequest = request
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
        for audio in preRoll { request.append(audio.buffer) }
        setState(.listening(activation))
        onTranscript?("")
    }

    private func handleRecognition(
        generation: UUID,
        transcript: String?,
        isFinal: Bool,
        errorMessage: String?
    ) {
        guard generation == sessionGeneration else { return }

        if let transcript, case .listening(let activation) = state {
            let visibleTranscript: String
            if activation == .wakeWord {
                visibleTranscript = commandAfterWakePhrase(in: transcript) ?? commandTranscript
            } else {
                visibleTranscript = transcript
            }
            commandTranscript = visibleTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
            onTranscript?(commandTranscript)
            if activation == .wakeWord, !commandTranscript.isEmpty {
                scheduleSilenceCompletion()
            }
        }

        if isFinal, case .listening(let activation) = state {
            completeCommand(activation: activation)
            return
        }

        if let errorMessage {
            let wasWakeListening = state == .wakeListening || state == .listening(.wakeWord)
            stopCapture(nextState: .idle)
            if wasWakeListening {
                failWakeWord(VoiceError.recognizerUnavailable)
            } else {
                onError?(errorMessage)
                if wakeListeningEnabled { scheduleWakeRestart() }
            }
        }
    }

    private func commandAfterWakePhrase(in transcript: String) -> String? {
        for phrase in [wakePhrase, "Sage"] {
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
        stopCapture(nextState: .processing)
        guard !command.isEmpty else {
            setState(.idle)
            onTranscript?("")
            scheduleWakeRestart()
            return
        }
        onCommand?(command, activation)
        scheduleWakeRestart()
    }

    private func appendToPreRoll(_ audio: VoiceAudioBuffer) {
        preRollBuffers.append(audio)
        preRollFrameCount += Int(audio.buffer.frameLength)
        let sampleRate = audio.buffer.format.sampleRate
        guard sampleRate > 0 else { return }
        let maximumFrames = Int(sampleRate * preRollDurationSeconds)
        while preRollFrameCount > maximumFrames, !preRollBuffers.isEmpty {
            preRollFrameCount -= Int(preRollBuffers.removeFirst().buffer.frameLength)
        }
    }

    private func clearPreRoll() {
        preRollBuffers.removeAll(keepingCapacity: true)
        preRollFrameCount = 0
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
            guard let self else { return }
            self.setState(.idle)
            self.resumeWakeListeningIfAuthorized()
        }
    }

    private func stopCapture(nextState: State) {
        wakeTimeoutTask?.cancel()
        wakeTimeoutTask = nil
        silenceTask?.cancel()
        silenceTask = nil

        if let engine = audioEngine {
            if inputTapInstalled { engine.inputNode.removeTap(onBus: 0) }
            engine.stop()
        }
        inputTapInstalled = false
        audioEngine = nil
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        sessionGeneration = UUID()
        clearPreRoll()
        setState(nextState)
    }

    private func failWakeWord(_ error: Error) {
        wakeListeningEnabled = false
        wakeListeningPaused = false
        wakeStartupGeneration = UUID()
        wakeStartupTask?.cancel()
        wakeStartupTask = nil
        wakeDetector?.pause()
        stopCapture(nextState: .idle)
        onError?(error.localizedDescription)
        onWakeWordUnavailable?()
    }

    private func setState(_ nextState: State) {
        guard state != nextState else { return }
        state = nextState
        onStateChange?(nextState)
    }

    private static func normalizedPhrase(_ phrase: String) -> String {
        let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.isEmpty ? "Hey Sage" : normalized
    }

    /// TCC does not guarantee that its authorization callback runs on the main queue.
    private nonisolated static func requestSpeechAuthorization() async
        -> SFSpeechRecognizerAuthorizationStatus
    {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// Audio is copied during the tap callback, then delivered to MainActor. The
    /// detector and Apple recognizer receive the same captured microphone stream.
    private nonisolated static func installAudioTap(
        on input: AVAudioInputNode,
        format: AVAudioFormat,
        deliver: @escaping @MainActor @Sendable (VoiceAudioBuffer) -> Void
    ) {
        input.installTap(onBus: 0, bufferSize: 1_024, format: format) { buffer, _ in
            guard let snapshot = VoiceAudioBuffer(copying: buffer) else { return }
            Task { @MainActor in deliver(snapshot) }
        }
    }

    /// Recognition results may arrive on an arbitrary queue; hop to MainActor before
    /// touching controller state.
    private nonisolated static func startRecognitionTask(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechAudioBufferRecognitionRequest,
        deliver: @escaping @MainActor @Sendable (String?, Bool, String?) -> Void
    ) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, error in
            let transcript = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let errorMessage = error?.localizedDescription
            Task { @MainActor in deliver(transcript, isFinal, errorMessage) }
        }
    }
}
