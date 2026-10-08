import Foundation

/// Local presentation/control routing only. The Rust core independently checks
/// every transcript revision and gates execution; these strings grant no access.
enum VoiceReflex {
    case stop, hold, correction(String)

    static func parse(_ text: String) -> VoiceReflex? {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.utf8.count <= 4096 else { return nil }
        let boundary = text.firstIndex { !$0.isASCII || !$0.isLetter } ?? text.endIndex
        let word = text[..<boundary].lowercased()
        let remainder = text[boundary...].trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: ",.!:")))
        switch word {
        case "stop": return .stop
        case "wait": return .hold
        case "hold" where remainder.lowercased() == "on" || remainder.lowercased().hasPrefix("on "): return .hold
        case "no", "actually": return .correction(remainder)
        default: return nil
        }
    }
}
