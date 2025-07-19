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
  
  s.source_files = 'ios/UniversalTranslationSDK/Sources/**/*.swift'
  s.resources = 'ios/UniversalTranslationSDK/Resources/**/*'
  
  s.frameworks = 'Foundation', 'CoreML', 'Compression', 'Network'
  s.ios.frameworks = 'UIKit'
  s.osx.frameworks = 'AppKit'
  
  s.dependency 'swift-log', '~> 1.5'
end