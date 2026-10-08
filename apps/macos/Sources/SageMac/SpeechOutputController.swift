@preconcurrency import AVFoundation
import Foundation

@MainActor
final class SpeechOutputController: NSObject, AVSpeechSynthesizerDelegate {
    private final class DelegateEvent: @unchecked Sendable {
        let utterance: AVSpeechUtterance
        init(_ utterance: AVSpeechUtterance) { self.utterance = utterance }
    }

    private let synthesizer = AVSpeechSynthesizer()
    private var pending: [AVSpeechUtterance] = []
    private var current: AVSpeechUtterance?
    private var streamFinished = false

    var onPlaybackStateChange: ((Bool) -> Void)?
    var onStreamDrained: (() -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func beginStream() {
        pending.removeAll(keepingCapacity: true)
        current = nil
        streamFinished = false
    }

    func enqueue(_ sentence: String) {
        let text = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        pending.append(AVSpeechUtterance(string: text))
        speakNextIfNeeded()
    }

    func finishStream() {
        streamFinished = true
        notifyDrainedIfNeeded()
    }

    func stopImmediately() {
        pending.removeAll()
        streamFinished = false
        let wasActive = current != nil || synthesizer.isSpeaking
        current = nil
        if wasActive { synthesizer.stopSpeaking(at: .immediate) }
        onPlaybackStateChange?(false)
    }

    private func speakNextIfNeeded() {
        guard current == nil, !pending.isEmpty else {
            notifyDrainedIfNeeded()
            return
        }
        let utterance = pending.removeFirst()
        current = utterance
        // Leaving `voice` unset selects the current macOS default voice.
        synthesizer.speak(utterance)
    }

    private func handleDidStart(_ utterance: AVSpeechUtterance) {
        guard current === utterance else { return }
        onPlaybackStateChange?(true)
    }

    private func handleDidFinish(_ utterance: AVSpeechUtterance) {
        guard current === utterance else { return }
        current = nil
        if pending.isEmpty {
            onPlaybackStateChange?(false)
        }
        speakNextIfNeeded()
    }

    private func handleDidCancel(_ utterance: AVSpeechUtterance) {
        guard current === utterance else { return }
        current = nil
        onPlaybackStateChange?(false)
        speakNextIfNeeded()
    }

    private func notifyDrainedIfNeeded() {
        guard streamFinished, current == nil, pending.isEmpty else { return }
        streamFinished = false
        onStreamDrained?()
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let event = DelegateEvent(utterance)
        Task { @MainActor [weak self, event] in self?.handleDidStart(event.utterance) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let event = DelegateEvent(utterance)
        Task { @MainActor [weak self, event] in self?.handleDidFinish(event.utterance) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let event = DelegateEvent(utterance)
        Task { @MainActor [weak self, event] in self?.handleDidCancel(event.utterance) }
    }
}
