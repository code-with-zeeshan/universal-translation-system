import XCTest
@testable import UniversalTranslationSDK

final class UniversalTranslationSDKTests: XCTestCase {
    func testTranslationClientInitialization() {
        let config = EncoderConfig(vocabUrl: "https://example.com/vocab")
        XCTAssertNotNil(config)
    }
}
