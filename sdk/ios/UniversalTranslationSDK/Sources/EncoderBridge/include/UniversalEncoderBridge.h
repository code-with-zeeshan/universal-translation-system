// ios/UniversalTranslationSDK/Sources/EncoderBridge/include/UniversalEncoderBridge.h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface UniversalEncoderBridge : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (BOOL)loadVocabulary:(NSString *)vocabPath error:(NSError **)error;

- (nullable NSData *)encodeText:(NSString *)text 
                     sourceLang:(NSString *)sourceLang 
                     targetLang:(NSString *)targetLang 
                          error:(NSError **)error;

- (NSArray<NSString *> *)getSupportedLanguages;
- (NSUInteger)getMemoryUsage;

@end

NS_ASSUME_NONNULL_END