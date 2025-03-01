import unittest
from scripts.data import utils


class TestPreprocessing(unittest.TestCase):
    """ Tests for different preprocessing functions """
    def setUp(self):
        self.corpus_dirty = [
            "aabb'.,1 ccddee99",
            "ccddeeffgg hJAKLm",
            "åchen."
        ]
        self.corpus_clean = [
            "aabb . one ccddeeninety nine",
            "ccddeeffgg hJAKLm",
            "achen."
        ]

        self.bio1_dirty = """
Birthday: July 21\n
Height: 176 cm\n
Blood type: A\n
Hobbies: soccer and reading\n
Favorite subject: science\n
Least favorite subject: art\n
Future dream: to be a cool guy\n
A popular and handsome third-year student who is madly in love with Ayako."""

        self.bio1_clean = "A popular and handsome third-year student who is madly in love with Ayako."

        self.bio2_dirty = """
Birthday - January 1\n
He is very cute."""
        self.bio2_clean = "He is very cute."

        self.bio3_dirty = """
Birthday\tJanuary 1\n
He is very cute."""
        self.bio3_clean = "He is very cute."

        self.bio4_dirty = "Birthday: January 1"

    def test_replace_text_numbers(self):

        self.assertEqual(utils.replace_text_numbers("Mayu Watanabe CG-3"),
                         "Mayu Watanabe CG-three")
        self.assertEqual(utils.replace_text_numbers("Pour Lui 13-sei"),
                         "Pour Lui thirteen-sei")
        self.assertEqual(utils.replace_text_numbers("Asuka Kuramochi the 9th"),
                         "Asuka Kuramochi the ninth")
        self.assertEqual(utils.replace_text_numbers("Yui Yokoyama the 7.5th"),
                         "Yui Yokoyama the seven point fifth")
        self.assertEqual(utils.replace_text_numbers("1.5"),
                         "one point five")
        self.assertEqual(utils.replace_text_numbers("02"),
                         "zero two")
        self.assertEqual(utils.replace_text_numbers(".01"),
                         ".zero one")
        self.assertEqual(utils.replace_text_numbers("The 03 is a lie"),
                         "The zero three is a lie")
    
    def test_clean_text_from_symbols(self):

        self.assertEqual(utils.clear_text_characters("Ángela Salas Larrazábal"),
                         "Angela Salas Larrazabal"),
        self.assertEqual(utils.clear_text_characters("Simo Häyhä"),
                         "Simo Hayha"),
        self.assertEqual(utils.clear_text_characters("Christine Waldegård"),
                         "Christine Waldegard"),
        self.assertEqual(utils.clear_text_characters("Selim Vergès"),
                         "Selim Verges"),
        self.assertEqual(utils.clear_text_characters("Padmé Amidala"),
                         "Padme Amidala"),
        self.assertEqual(utils.clear_text_characters("Pierre Tempête de Neige"),
                         "Pierre Tempete de Neige"),
        self.assertEqual(utils.clear_text_characters("Chloë Maxwell"),
                         "Chloe Maxwell"),
        self.assertEqual(utils.clear_text_characters("Bernardo Dión"),
                         "Bernardo Dion"),
        self.assertEqual(utils.clear_text_characters("Gérôme Hongou"),
                         "Gerome Hongou"),
        self.assertEqual(utils.clear_text_characters("Arad Mölders"),
                         "Arad Molders"),
        self.assertEqual(utils.clear_text_characters("Tor Nørretranders"),
                         "Tor Norretranders"),
        self.assertEqual(utils.clear_text_characters("Jürgen von Klügel"),
                         "Jurgen von Klugel"),
        self.assertEqual(utils.clear_text_characters("Œlaf"),
                         "OElaf"),
        self.assertEqual(utils.clear_text_characters("Daša Urban"),
                         "Dasa Urban")
        self.assertEqual(utils.clear_text_characters("02,';'1"),
                         "zero two one")
        self.assertEqual(utils.clear_text_characters("Åll your 1.2 bases are. SO bel']ong to-us 13"),
                         "All your one point two bases are. SO bel ong to-us thirteen")

    def test_clear_corpus(self):
        self.assertListEqual(utils.clear_corpus_characters(self.corpus_dirty, 1), self.corpus_clean)


    def test_clean_bio(self):
        self.assertEqual(utils.clean_bio(self.bio1_dirty), self.bio1_clean)
        self.assertEqual(utils.clean_bio(self.bio2_dirty), self.bio2_clean)
        self.assertEqual(utils.clean_bio(self.bio3_dirty), self.bio3_clean)
        if False: # TODO: an edge case, can't design a regex for this just yet.
            self.assertEqual(utils.clean_bio(self.bio4_dirty), "")
        