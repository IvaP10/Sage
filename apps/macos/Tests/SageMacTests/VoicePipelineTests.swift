import Foundation
import XCTest
@testable import SageMac

final class VoicePipelineTests: XCTestCase {
    func testCumulativeAnswerChunksSpeakCompleteSentencesOnceAndFlushFinalText() {
        var buffer = ResponseSentenceBuffer()

        XCTAssertEqual(buffer.appendCumulative("Hello there. "), ["Hello there."])
        XCTAssertEqual(buffer.appendCumulative("Hello there. "), [])
        XCTAssertEqual(buffer.appendCumulative("Hello there. The answer is still"), [])
        XCTAssertEqual(
            buffer.finishCumulative("Hello there. The answer is still coming."),
            ["The answer is still coming."]
        )
    }

    func testSentenceBoundaryKeepsClosingQuoteAndWaitsForFollowingSpace() {
        var buffer = ResponseSentenceBuffer()
        XCTAssertEqual(buffer.appendCumulative("She said, \"Ready?\""), [])
        XCTAssertEqual(buffer.appendCumulative("She said, \"Ready?\" Next"), ["She said, \"Ready?\""])
        XCTAssertEqual(buffer.finishCumulative("She said, \"Ready?\" Next."), ["Next."])
    }

    func testWakePhraseMatcherUsesWholeTokenSequences() {
        XCTAssertTrue(WakePhraseMatcher.contains("Hey Sage", in: "Hey, Sage! Open the file."))
        XCTAssertTrue(WakePhraseMatcher.contains("Sage", in: "Sage, open the file."))
        XCTAssertFalse(WakePhraseMatcher.contains("Sage", in: "Make this sagely."))
        XCTAssertFalse(WakePhraseMatcher.contains("Hey Sage", in: "Hey, wise Sage."))
    }

    func testWakePhraseRejectsNonEnglishCharacters() throws {
        XCTAssertFalse(WakePhraseMatcher.isValidPhrase("Café Sage"))
        XCTAssertFalse(WakePhraseMatcher.isValidPhrase(""))
        XCTAssertFalse(WakePhraseMatcher.isValidPhrase("one two three four five six seven"))
        XCTAssertTrue(WakePhraseMatcher.isValidPhrase("Hey, Sage!"))
    }

    @MainActor
    func testWakeListeningIsOffWhenVoiceInputControllerStarts() {
        let controller = VoiceInputController()
        XCTAssertEqual(controller.state, .idle)
        XCTAssertFalse(controller.wakeListeningEnabled)
        controller.stop()
    }

    func testLocalWakeDetectorLoadsNativeRecognizerWithoutStartingMicrophone() async throws {
        let detector = try await LocalWakeWordDetector.load(phrase: "Hey Sage")
        XCTAssertEqual(detector.phrase, "Hey Sage")
        detector.pause()
    }
}
