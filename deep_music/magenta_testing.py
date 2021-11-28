import ctypes.util

# orig_ctypes_util_find_library = ctypes.util.find_library
# def proxy_find_library(lib):
#   if lib == 'fluidsynth':
#     return 'libfluidsynth.so.1'
#   else:
#     return orig_ctypes_util_find_library(lib)
# ctypes.util.find_library = proxy_find_library


import fluidsynth
import magenta
import note_seq
import tensorflow

print('🎉 Done!')
print(magenta.__version__)
print(tensorflow.__version__)







from note_seq.protobuf import music_pb2




# print('Downloading model bundle. This will take less than a minute...')
# note_seq.notebook_utils.download_bundle('basic_rnn.mag', 'deep_music/data/magenta_models/')

# # Import dependencies.
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

# # Initialize the model.
# print("Initializing Melody RNN...")
MODEL_PATH = 'deep_music/data/magenta_models/basic_rnn.mag'
SAVE_DIRECTORY = "deep_music/data/results/"
RAW_DATA_PATH = "raw_data/snes/"

class PreTrainedRnn():

    def __init__(self, path_to_model):
        bundle = sequence_generator_bundle.read_bundle_file(
            path_to_model)
        generator_map = melody_rnn_sequence_generator.get_generator_map()
        self.melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
        self.melody_rnn.initialize()
        self.fdir = SAVE_DIRECTORY

    def melody_to_seq(self, melody_path):

        melody = note_seq.midi_file_to_melody(melody_path)

        return melody.to_sequence()

    def generate_new_melody(self, seq_len, temp, input_seq, midi_name):
        """
        seq_len : int : how long you want the sequence to be
        temp : float : > 0 : how random the sequence will be
        input_seq : sequence of note_seq object
        midi_name : midi_name the file name you want saved
        """
        num_steps = seq_len  # change this for shorter or longer sequences


        temperature = temp  # the higher the temperature the more random the sequence.

        # Set the start time to begin on the next step after the last note ends.
        last_end_time = (max(
            n.end_time for n in input_seq.notes) if input_seq.notes else 0)


        qpm = input_seq.tempos[0].qpm
        seconds_per_step = 60.0 / qpm / self.melody_rnn.steps_per_quarter
        total_seconds = num_steps * seconds_per_step

        generator_options = generator_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = temperature
        generate_section = generator_options.generate_sections.add(
            start_time=last_end_time + seconds_per_step, end_time=total_seconds)

        # Ask the model to continue the sequence.
        sequence = self.melody_rnn.generate(input_seq, generator_options)

        return note_seq.sequence_proto_to_midi_file(
            sequence, self.fdir + f"{midi_name}")



if __name__ == "__main__":
    pre_trained_model = PreTrainedRnn(MODEL_PATH)
    mario = pre_trained_model.melody_to_seq(RAW_DATA_PATH + "SMW-Underwater_XG.mid")
    pre_trained_model.generate_new_melody(500, 1, mario, "test_midi.mid")

# dkc_seq.set_length(24)



# print('🎉 Done!')




# # Import dependencies.
# from magenta.models.music_vae import configs
# from magenta.models.music_vae.trained_model import TrainedModel

# # Initialize the model.
# print("Initializing Music VAE...")
# music_vae = TrainedModel(
#       configs.CONFIG_MAP['cat-mel_2bar_big'],
#       batch_size=4,
#       checkpoint_dir_or_path='/content/mel_2bar_big.ckpt')

# print('🎉 Done!')




# print(dkc_seq.to_sequence())

# mario_seq = note_seq.midi_file_to_melody("raw_data/snes/SMW-Underwater_XG.mid")
# mario_seq.to_sequence()



# # # Model options. Change these to get different generated sequences!

# input_sequence = mario_seq.to_sequence() # change this to teapot if you want



## Bokeh Methods
# note_seq.plot_sequence(sequence)
# note_seq.play_sequence(sequence, synth=note_seq.fluidsynth)









# # We're going to interpolate between the Twinkle Twinkle Little Star
# # NoteSequence we defined in the first section, and one of the generated
# # sequences from the previous VAE example

# # How many sequences, including the start and end ones, to generate.
# num_steps = 16

# # This gives us a list of sequences.
# note_sequences = music_vae.interpolate(
#       mario_seq.to_sequence(),
#       dkc_seq.to_sequence(),
#       num_steps=num_steps,
#       length=32)

# # Concatenate them into one long sequence, with the start and
# # end sequences at each end.
# interp_seq = note_seq.sequences_lib.concatenate_sequences(note_sequences)

# note_seq.play_sequence(interp_seq, synth=note_seq.synthesize)
# note_seq.plot_sequence(interp_seq)
