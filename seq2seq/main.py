# -*- coding: utf-8 -*-

from seq2seq_translation_tutorial import EncoderRNN, AttnDecoderRNN, trainIters, evaluateRandomly, input_lang, output_lang, device

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(
    hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)
