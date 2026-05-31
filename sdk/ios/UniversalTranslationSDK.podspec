# ios/UniversalTranslationSDK.podspec

Pod::Spec.new do |s|
  s.name             = 'UniversalTranslationSDK'
  s.version          = '1.0.0'
  s.summary          = 'Universal Translation System SDK for iOS'
  s.description      = <<-DESC
    iOS SDK for the Universal Translation System with on-device encoding 
    and cloud decoding. Supports 20+ languages with dynamic vocabulary loading.
  DESC
  
  s.homepage         = 'https://github.com/yourusername/universal-translation-system'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Your Name' => 'your.email@example.com' }
  s.source           = { :git => 'https://github.com/yourusername/universal-translation-system.git', :tag => s.version.to_s }
  
  s.ios.deployment_target = '15.0'
  s.osx.deployment_target = '12.0'
  s.watchos.deployment_target = '8.0'
  s.tvos.deployment_target = '15.0'
  
  s.swift_version = '5.7'
  
  s.source_files = 'ios/UniversalTranslationSDK/Sources/**/*.{swift,h,m,mm}'
  s.public_header_files = 'ios/UniversalTranslationSDK/Sources/EncoderBridge/include/*.h'
  s.resources = 'ios/UniversalTranslationSDK/Resources/**/*'
  
  s.frameworks = 'Foundation', 'CoreML', 'Compression', 'Network'
  s.ios.frameworks = 'UIKit'
  s.osx.frameworks = 'AppKit'
  
  s.libraries = 'c++'
  
  # Dependencies
  s.dependency 'MessagePack.swift', '~> 4.0'
  s.dependency 'SWCompression', '~> 4.8'
  
  # If using ONNX Runtime
  s.ios.vendored_frameworks = 'Frameworks/onnxruntime.xcframework'
  
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'OTHER_CPLUSPLUSFLAGS' => '-std=c++17 -stdlib=libc++',
    'HEADER_SEARCH_PATHS' => '$(PODS_ROOT)/../../encoder_core/include'
  }
end