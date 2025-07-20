require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
folly_compiler_flags = '-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -Wno-comma -Wno-shorten-64-to-32'

Pod::Spec.new do |s|
  s.name         = "universal-translation-sdk"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "15.0" }
  s.source       = { :git => "https://github.com/yourusername/universal-translation-system.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm,swift}"
  s.requires_arc = true

  s.dependency "React-Core"
  
  # Include the native SDK files directly instead of depending on a non-existent pod
  # Copy the native SDK files into the project
  s.preserve_paths = 'ios/UniversalTranslationSDK/**/*'
  
  # Link with required frameworks
  s.frameworks = 'Foundation', 'CoreML', 'Compression', 'Network'
  
  # Swift version
  s.swift_version = '5.7'
  
  # If using local Universal Translation SDK files
  s.vendored_frameworks = 'ios/Frameworks/UniversalTranslationSDK.xcframework' if File.exist?('ios/Frameworks/UniversalTranslationSDK.xcframework')
  
  # Don't install the dependencies when we run `pod install` in the old architecture.
  if ENV['RCT_NEW_ARCH_ENABLED'] == '1' then
    s.compiler_flags = folly_compiler_flags + " -DRCT_NEW_ARCH_ENABLED=1"
    s.pod_target_xcconfig    = {
        "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/boost\"",
        "OTHER_CPLUSPLUSFLAGS" => "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1",
        "CLANG_CXX_LANGUAGE_STANDARD" => "c++17"
    }
    s.dependency "React-Codegen"
    s.dependency "RCT-Folly"
    s.dependency "RCTRequired"
    s.dependency "RCTTypeSafety"
    s.dependency "ReactCommon/turbomodule/core"
  else
    s.pod_target_xcconfig = {
      "SWIFT_VERSION" => "5.7",
      "DEFINES_MODULE" => "YES",
      "SWIFT_OPTIMIZATION_LEVEL" => "-Owholemodule"
    }
  end
  
  # Add a script phase to copy native SDK files if needed
  s.script_phases = [
    {
      :name => 'Copy Universal Translation SDK',
      :script => 'cp -R "${PODS_TARGET_SRCROOT}/../../ios/UniversalTranslationSDK/Sources/"* "${PODS_TARGET_SRCROOT}/ios/" 2>/dev/null || :',
      :execution_position => :before_compile
    }
  ]
end