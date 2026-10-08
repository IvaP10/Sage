import Foundation

/// Reads only the bounded universal-header table. Security.framework validates
/// the executable bytes and signatures at the resulting offsets.
enum MachOSlices {
    static func offsets(header: Data, fileSize: UInt64) throws -> [UInt64] {
        let bytes = Array(header)
        func number(_ start: Int, _ width: Int) throws -> UInt64 {
            guard start >= 0, width <= 8, start + width <= bytes.count else { throw invalid() }
            return bytes[start..<(start + width)].reduce(0) { ($0 << 8) | UInt64($1) }
        }
        let magic = try number(0, 4)
        if [0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe].contains(magic) {
            guard fileSize >= 32 else { throw invalid() }
            return [0]
        }
        // Universal tables are big-endian on disk (mach-o/fat.h).
        guard magic == 0xcafebabe || magic == 0xcafebabf else { throw invalid() }
        let count = try number(4, 4)
        guard (1...32).contains(count) else { throw invalid() }
        let width = magic == 0xcafebabf ? 8 : 4
        let entrySize = width == 8 ? 32 : 20
        let tableEnd = 8 + Int(count) * entrySize
        guard tableEnd <= bytes.count, UInt64(tableEnd) <= fileSize else { throw invalid() }
        var ranges: [Range<UInt64>] = []
        for index in 0..<Int(count) {
            let start = 8 + index * entrySize
            let offset = try number(start + 8, width)
            let size = try number(start + 8 + width, width)
            guard offset >= UInt64(tableEnd), size >= 32, offset <= fileSize, size <= fileSize - offset else { throw invalid() }
            let range = offset..<(offset + size)
            guard !ranges.contains(where: { $0.overlaps(range) }) else { throw invalid() }
            ranges.append(range)
        }
        return ranges.map(\.lowerBound)
    }
    private static func invalid() -> Error { SageClientError.protocolError("Unsupported or malformed Mach-O architecture table") }
}
