import Foundation

struct ResponseSentenceBuffer {
    private(set) var cumulativeText = ""
    private var pendingText = ""

    /// The IPC stream sends the whole answer prefix each time. Only the newly
    /// appended suffix is considered for speech, so repeated prefixes stay silent.
    mutating func appendCumulative(_ text: String) -> [String] {
        let suffix: String
        if cumulativeText.isEmpty {
            suffix = text
        } else if text.hasPrefix(cumulativeText) {
            suffix = String(text.dropFirst(cumulativeText.count))
        } else {
            // A rewritten prefix cannot be spoken safely after earlier sentences
            // have already been heard. Keep the last stable version.
            return []
        }
        cumulativeText = text
        pendingText += suffix
        return takeCompleteSentences()
    }

    mutating func finishCumulative(_ text: String) -> [String] {
        var sentences = appendCumulative(text)
        let remainder = pendingText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !remainder.isEmpty { sentences.append(remainder) }
        pendingText = ""
        return sentences
    }

    private mutating func takeCompleteSentences() -> [String] {
        var sentences: [String] = []
        while let range = firstSentenceBoundary(in: pendingText) {
            let sentence = String(pendingText[..<range.sentenceEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { sentences.append(sentence) }
            pendingText = String(pendingText[range.afterWhitespace...])
        }
        return sentences
    }

    private func firstSentenceBoundary(in text: String) -> (sentenceEnd: String.Index, afterWhitespace: String.Index)? {
        let closingPunctuation: Set<Character> = ["\"", "'", "”", "’", ")", "]", "}"]
        var index = text.startIndex
        while index < text.endIndex {
            guard ".!?".contains(text[index]) else {
                index = text.index(after: index)
                continue
            }
            var sentenceEnd = text.index(after: index)
            while sentenceEnd < text.endIndex, closingPunctuation.contains(text[sentenceEnd]) {
                sentenceEnd = text.index(after: sentenceEnd)
            }
            guard sentenceEnd < text.endIndex, text[sentenceEnd].isWhitespace else {
                index = text.index(after: index)
                continue
            }
            var afterWhitespace = sentenceEnd
            while afterWhitespace < text.endIndex, text[afterWhitespace].isWhitespace {
                afterWhitespace = text.index(after: afterWhitespace)
            }
            return (sentenceEnd, afterWhitespace)
        }
        return nil
    }
}
