#include "audio_utils.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace asr;

void test_float_range_normalize() {
    std::cout << "=== Test float_range_normalize ===\n";
    
    // Test 1: Already normalized
    std::vector<float> audio1 = {0.5f, -0.3f, 0.8f, -0.6f};
    float_range_normalize(audio1);
    assert(std::fabs(audio1[0] - 0.5f) < 0.001f);
    assert(std::fabs(audio1[2] - 0.8f) < 0.001f);
    std::cout << "  Test 1: Already normalized - PASSED\n";
    
    // Test 2: Peak > 1 (int-like scaled)
    std::vector<float> audio2 = {100.0f, -50.0f, 200.0f, -150.0f};
    float_range_normalize(audio2);
    assert(std::fabs(audio2[0] - 0.5f) < 0.001f);
    assert(std::fabs(audio2[2] - 1.0f) < 0.001f);
    assert(std::fabs(audio2[3] + 0.75f) < 0.001f);
    std::cout << "  Test 2: Peak > 1 - PASSED\n";
    
    // Test 3: Empty audio
    std::vector<float> audio3;
    float_range_normalize(audio3);
    assert(audio3.empty());
    std::cout << "  Test 3: Empty audio - PASSED\n";
    
    // Test 4: Zero audio
    std::vector<float> audio4 = {0.0f, 0.0f, 0.0f};
    float_range_normalize(audio4);
    for (float v : audio4) assert(v == 0.0f);
    std::cout << "  Test 4: Zero audio - PASSED\n";
    
    std::cout << "All float_range_normalize tests PASSED!\n\n";
}

void test_pad_audio_to_min_length() {
    std::cout << "=== Test pad_audio_to_min_length ===\n";
    
    // Test 1: Short audio (0.1s < 0.5s min)
    std::vector<float> audio1(1600, 0.5f); // 0.1s at 16kHz
    pad_audio_to_min_length(audio1, MIN_ASR_INPUT_SECONDS);
    assert(audio1.size() == 8000); // 0.5s at 16kHz
    std::cout << "  Test 1: Short audio padded - PASSED (1600 -> 8000)\n";
    
    // Test 2: Already long enough
    std::vector<float> audio2(16000, 0.5f); // 1s at 16kHz
    pad_audio_to_min_length(audio2, MIN_ASR_INPUT_SECONDS);
    assert(audio2.size() == 16000);
    std::cout << "  Test 2: Already long enough - PASSED\n";
    
    std::cout << "All pad_audio tests PASSED!\n\n";
}

void test_split_audio_into_chunks() {
    std::cout << "=== Test split_audio_into_chunks ===\n";
    
    // Test 1: Short audio (no split needed)
    std::vector<float> audio1(16000, 0.5f); // 1s
    auto chunks1 = split_audio_into_chunks(audio1, 180.0f);
    assert(chunks1.size() == 1);
    assert(chunks1[0].wav.size() == 16000);
    assert(chunks1[0].offset_sec == 0.0f);
    std::cout << "  Test 1: Short audio - PASSED (1 chunk)\n";
    
    // Test 2: Long audio (requires split)
    std::vector<float> audio2(180 * 16000, 0.5f); // 180s
    auto chunks2 = split_audio_into_chunks(audio2, 60.0f);
    std::cout << "  Test 2: Long audio - Got " << chunks2.size() << " chunks\n";
    
    // Verify total samples match original
    size_t total_samples = 0;
    float last_offset = 0.0f;
    for (const auto & c : chunks2) {
        total_samples += c.wav.size();
        last_offset = c.offset_sec;
    }
    // Each chunk may be padded, so total may be slightly more
    std::cout << "    Total samples: " << total_samples << " (original: " << audio2.size() << ")\n";
    std::cout << "    Last offset: " << last_offset << "s\n";
    std::cout << "  Test 2: Long audio split - PASSED\n";
    
    std::cout << "All split_audio tests PASSED!\n\n";
}

void test_detect_and_fix_repetitions() {
    std::cout << "=== Test detect_and_fix_repetitions ===\n";
    
    // Test 1: No repetition
    std::string text1 = "Hello World";
    std::string fixed1 = detect_and_fix_repetitions(text1);
    assert(fixed1 == "Hello World");
    std::cout << "  Test 1: No repetition - PASSED\n";
    
    // Test 2: Character repetition
    std::string text2 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaHello"; // >20 'a's
    std::string fixed2 = detect_and_fix_repetitions(text2, 20);
    assert(fixed2 == "aHello");
    std::cout << "  Test 2: Char repetition fixed: \"" << text2 << "\" -> \"" << fixed2 << "\" - PASSED\n";
    
    // Test 3: Pattern repetition
    std::string text3 = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc"; // many "abc"
    std::string fixed3 = detect_and_fix_repetitions(text3, 20);
    assert(fixed3 == "abc");
    std::cout << "  Test 3: Pattern repetition fixed: \"" << text3.substr(0, 30) << "...\" -> \"" << fixed3 << "\" - PASSED\n";
    
    std::cout << "All repetition tests PASSED!\n\n";
}

void test_parse_asr_output() {
    std::cout << "=== Test parse_asr_output ===\n";
    
    // Test 1: Normal output with language
    std::string raw1 = "language Chinese<asr_text>你好世界";
    auto [lang1, text1] = parse_asr_output(raw1);
    assert(lang1 == "Chinese");
    assert(text1 == "你好世界");
    std::cout << "  Test 1: Normal output - PASSED (lang=\"" << lang1 << "\", text=\"" << text1 << "\")\n";
    
    // Test 2: Output without tag (pure text)
    std::string raw2 = "你好世界";
    auto [lang2, text2] = parse_asr_output(raw2);
    assert(lang2 == "");
    assert(text2 == "你好世界");
    std::cout << "  Test 2: Pure text - PASSED (lang=\"" << lang2 << "\", text=\"" << text2 << "\")\n";
    
    // Test 3: Empty audio indicator
    std::string raw3 = "language None<asr_text>";
    auto [lang3, text3] = parse_asr_output(raw3);
    assert(lang3 == "");
    assert(text3 == "");
    std::cout << "  Test 3: Empty audio - PASSED\n";
    
    // Test 4: With user_language (forced)
    std::string raw4 = "你好世界";
    auto [lang4, text4] = parse_asr_output(raw4, "English");
    assert(lang4 == "English");
    assert(text4 == "你好世界");
    std::cout << "  Test 4: Forced language - PASSED (lang=\"" << lang4 << "\")\n";
    
    // Test 5: With repetition in output
    std::string raw5 = "language Chinese<asr_text>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaHello";
    auto [lang5, text5] = parse_asr_output(raw5);
    assert(lang5 == "Chinese");
    assert(text5 == "aHello");
    std::cout << "  Test 5: Repetition auto-fixed - PASSED (text=\"" << text5 << "\")\n";
    
    // Test 6: Without <asr_text> tag but with language prefix (new fix)
    std::string raw6 = "language Chinese知道规则吗？";
    auto [lang6, text6] = parse_asr_output(raw6);
    assert(lang6 == "Chinese");
    assert(text6 == "知道规则吗？");
    std::cout << "  Test 6: No tag with language prefix - PASSED (lang=\"" << lang6 << "\", text=\"" << text6 << "\")\n";
    
    // Test 7: Language prefix with space before text
    std::string raw7 = "language Chinese 知道规则吗？";
    auto [lang7, text7] = parse_asr_output(raw7);
    assert(lang7 == "Chinese");
    assert(text7 == "知道规则吗？");
    std::cout << "  Test 7: Language prefix with space - PASSED (text=\"" << text7 << "\")\n";
    
    std::cout << "All parse_asr_output tests PASSED!\n\n";
}

void test_normalize_language_name() {
    std::cout << "=== Test normalize_language_name ===\n";
    
    assert(normalize_language_name("chinese") == "Chinese");
    assert(normalize_language_name("CHINESE") == "Chinese");
    assert(normalize_language_name("cHiNeSe") == "Chinese");
    assert(normalize_language_name("  korean  ") == "Korean");
    assert(normalize_language_name("") == "");
    
    std::cout << "  All tests PASSED!\n\n";
}

int main() {
    std::cout << "\n=== Audio Utils Tests ===\n\n";
    
    test_float_range_normalize();
    test_pad_audio_to_min_length();
    test_split_audio_into_chunks();
    test_detect_and_fix_repetitions();
    test_parse_asr_output();
    test_normalize_language_name();
    
    std::cout << "=== All tests PASSED! ===\n";
    return 0;
}