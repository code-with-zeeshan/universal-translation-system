# android/UniversalTranslationSDK/proguard-rules.pro

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep translation encoder classes
-keep class com.universaltranslation.encoder.** { *; }

# Keep data classes
-keep class com.universaltranslation.encoder.VocabularyPack { *; }
-keep class com.universaltranslation.encoder.TranslationResponse { *; }
-keep class com.universaltranslation.encoder.TranslationResult { *; }
-keep class com.universaltranslation.encoder.TranslationResult$* { *; }

# ONNX Runtime
-keep class ai.onnxruntime.** { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-keepnames class okhttp3.internal.publicsuffix.PublicSuffixDatabase

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.** { *; }

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}