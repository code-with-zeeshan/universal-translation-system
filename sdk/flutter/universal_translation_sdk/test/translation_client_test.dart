import 'package:flutter_test/flutter_test.dart';
import 'package:universal_translation_sdk/universal_translation_sdk.dart';

void main() {
  test('TranslationClient can be instantiated', () {
    final client = TranslationClient();
    expect(client, isNotNull);
  });
}
