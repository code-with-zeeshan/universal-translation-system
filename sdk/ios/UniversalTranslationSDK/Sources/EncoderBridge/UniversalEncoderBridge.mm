// ios/UniversalTranslationSDK/Sources/EncoderBridge/UniversalEncoderBridge.mm

#import "UniversalEncoderBridge.h"
#include "universal_encoder.h"
#include <memory>
#include <string>
#include <vector>

NSString *const UniversalEncoderErrorDomain = @"com.universaltranslation.encoder";

typedef NS_ENUM(NSInteger, UniversalEncoderError) {
    UniversalEncoderErrorInitialization = 1000,
    UniversalEncoderErrorVocabularyLoad = 1001,
    UniversalEncoderErrorEncoding = 1002,
    UniversalEncoderErrorInvalidInput = 1003
};

@interface UniversalEncoderBridge () {
    std::unique_ptr<UniversalTranslation::UniversalEncoder> _encoder;
}
@end

@implementation UniversalEncoderBridge

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
    self = [super init];
    if (self) {
        try {
            std::string path = [modelPath UTF8String];
            _encoder = std::make_unique<UniversalTranslation::UniversalEncoder>(path);
        } catch (const std::exception& e) {
            if (error) {
                *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                             code:UniversalEncoderErrorInitialization
                                         userInfo:@{
                    NSLocalizedDescriptionKey: @"Failed to initialize encoder",
                    NSLocalizedFailureReasonErrorKey: [NSString stringWithUTF8String:e.what()]
                }];
            }
            return nil;
        }
    }
    return self;
}

- (BOOL)loadVocabulary:(NSString *)vocabPath error:(NSError **)error {
    if (!_encoder) {
        if (error) {
            *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                         code:UniversalEncoderErrorVocabularyLoad
                                     userInfo:@{NSLocalizedDescriptionKey: @"Encoder not initialized"}];
        }
        return NO;
    }
    
    try {
        std::string path = [vocabPath UTF8String];
        bool success = _encoder->loadVocabulary(path);
        
        if (!success) {
            if (error) {
                *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                             code:UniversalEncoderErrorVocabularyLoad
                                         userInfo:@{NSLocalizedDescriptionKey: @"Failed to load vocabulary"}];
            }
            return NO;
        }
        
        return YES;
    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                         code:UniversalEncoderErrorVocabularyLoad
                                     userInfo:@{
                NSLocalizedDescriptionKey: @"Failed to load vocabulary",
                NSLocalizedFailureReasonErrorKey: [NSString stringWithUTF8String:e.what()]
            }];
        }
        return NO;
    }
}

- (nullable NSData *)encodeText:(NSString *)text 
                     sourceLang:(NSString *)sourceLang 
                     targetLang:(NSString *)targetLang 
                          error:(NSError **)error {
    if (!_encoder) {
        if (error) {
            *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                         code:UniversalEncoderErrorEncoding
                                     userInfo:@{NSLocalizedDescriptionKey: @"Encoder not initialized"}];
        }
        return nil;
    }
    
    if (text.length == 0) {
        if (error) {
            *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                         code:UniversalEncoderErrorInvalidInput
                                     userInfo:@{NSLocalizedDescriptionKey: @"Input text is empty"}];
        }
        return nil;
    }
    
    try {
        std::string textStr = [text UTF8String];
        std::string sourceStr = [sourceLang UTF8String];
        std::string targetStr = [targetLang UTF8String];
        
        std::vector<uint8_t> encoded = _encoder->encode(textStr, sourceStr, targetStr);
        
        return [NSData dataWithBytes:encoded.data() length:encoded.size()];
        
    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:UniversalEncoderErrorDomain
                                         code:UniversalEncoderErrorEncoding
                                     userInfo:@{
                NSLocalizedDescriptionKey: @"Encoding failed",
                NSLocalizedFailureReasonErrorKey: [NSString stringWithUTF8String:e.what()]
            }];
        }
        return nil;
    }
}

- (NSArray<NSString *> *)getSupportedLanguages {
    if (!_encoder) {
        return @[];
    }
    
    std::vector<std::string> languages = _encoder->getSupportedLanguages();
    NSMutableArray<NSString *> *result = [NSMutableArray arrayWithCapacity:languages.size()];
    
    for (const auto& lang : languages) {
        [result addObject:[NSString stringWithUTF8String:lang.c_str()]];
    }
    
    return result;
}

- (NSUInteger)getMemoryUsage {
    if (!_encoder) {
        return 0;
    }
    
    return static_cast<NSUInteger>(_encoder->getMemoryUsage());
}

@end