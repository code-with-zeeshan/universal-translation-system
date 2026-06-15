package com.universaltranslation.encoder;

import org.junit.Test;
import static org.junit.Assert.*;

public class TranslationEncoderTest {
    @Test
    public void testInitialization() {
        TranslationEncoder encoder = new TranslationEncoder();
        assertNotNull(encoder);
    }
}
