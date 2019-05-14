# requirement: want to make sure that text is processed correctly

import unittest
from src import train_annotation_generator

class TestTrainAnnotationGenerator(unittest.TestCase):
    def test_clean_text(self):
        ag = train_annotation_generator.Seq2Seq_Train_Annotation_Generator(model_name='base',
                                            num_epochs=100,
                                            latent_dim=256,
                                            optimizer='adam')
        result = ag._clean_text(['Hello there!',
                                "You're sure this handles contractions well?",
                                'well what about years? I think 1999 should be ok!!!',
                                "what about... elipses...?! How will that be handled?",
                                'maybe     extra spaces will be taken care of too...'])

        self.assertEqual(result[0], 'hello there!')
        self.assertEqual(result[1], "you're sure this handles contractions well?")
        self.assertEqual(result[2], 'well what about years? i think 1999 should be ok!!!')
        self.assertEqual(result[3], "what about... elipses...?! how will that be handled?")
        self.assertEqual(result[4], 'maybe extra spaces will be taken care of too...')


if __name__ == '__main__':
    unittest.main()
