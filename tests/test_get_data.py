import unittest
from scripts.data import download, preprocessing


class TestDownload(unittest.TestCase):
    """ Tests for link and character page download functions """
    def setUp(self):
        self.descriptions_dirty = [
            "Here's a description.\n(Source: Somewebsite)",
            "\n\nHere's another description.\n\nHere's one more.",
            "       Here's the last description.\n\n           ",
            "No biography written."
        ]
        self.descriptions_clean = [
            "Here's a description.",
            "Here's another description.\nHere's one more.",
            "Here's the last description.",
            ""
        ]

    def test_beautify(self):
        descriptions_beautified = [download.beautify_bio(bio) for bio in self.descriptions_dirty]
        self.assertEqual(descriptions_beautified, self.descriptions_clean, 'Bios not beautified as expected.')


class TestPreprocess(unittest.TestCase):
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

    def test_replace_text_numbers(self):

        self.assertEqual(preprocessing.replace_text_numbers("Mayu Watanabe CG-3"),
                         "Mayu Watanabe CG-three")
        self.assertEqual(preprocessing.replace_text_numbers("Pour Lui 13-sei"),
                         "Pour Lui thirteen-sei")
        self.assertEqual(preprocessing.replace_text_numbers("Asuka Kuramochi the 9th"),
                         "Asuka Kuramochi the ninth")
        self.assertEqual(preprocessing.replace_text_numbers("Yui Yokoyama the 7.5th"),
                         "Yui Yokoyama the seven point fifth")
        self.assertEqual(preprocessing.replace_text_numbers("1.5"),
                         "one point five")
        self.assertEqual(preprocessing.replace_text_numbers("02"),
                         "zero two")
        self.assertEqual(preprocessing.replace_text_numbers(".01"),
                         ".zero one")
        self.assertEqual(preprocessing.replace_text_numbers("The 03 is a lie"),
                         "The zero three is a lie")
    
    def test_clean_text_from_symbols(self):

        self.assertEqual(preprocessing.clear_text_characters("Ángela Salas Larrazábal"),
                         "Angela Salas Larrazabal"),
        self.assertEqual(preprocessing.clear_text_characters("Simo Häyhä"),
                         "Simo Hayha"),
        self.assertEqual(preprocessing.clear_text_characters("Christine Waldegård"),
                         "Christine Waldegard"),
        self.assertEqual(preprocessing.clear_text_characters("Selim Vergès"),
                         "Selim Verges"),
        self.assertEqual(preprocessing.clear_text_characters("Padmé Amidala"),
                         "Padme Amidala"),
        self.assertEqual(preprocessing.clear_text_characters("Pierre Tempête de Neige"),
                         "Pierre Tempete de Neige"),
        self.assertEqual(preprocessing.clear_text_characters("Chloë Maxwell"),
                         "Chloe Maxwell"),
        self.assertEqual(preprocessing.clear_text_characters("Bernardo Dión"),
                         "Bernardo Dion"),
        self.assertEqual(preprocessing.clear_text_characters("Gérôme Hongou"),
                         "Gerome Hongou"),
        self.assertEqual(preprocessing.clear_text_characters("Arad Mölders"),
                         "Arad Molders"),
        self.assertEqual(preprocessing.clear_text_characters("Tor Nørretranders"),
                         "Tor Norretranders"),
        self.assertEqual(preprocessing.clear_text_characters("Jürgen von Klügel"),
                         "Jurgen von Klugel"),
        self.assertEqual(preprocessing.clear_text_characters("Œlaf"),
                         "OElaf"),
        self.assertEqual(preprocessing.clear_text_characters("Daša Urban"),
                         "Dasa Urban")
        self.assertEqual(preprocessing.clear_text_characters("02,';'1"),
                         "zero two one")
        self.assertEqual(preprocessing.clear_text_characters("Åll your 1.2 bases are. SO bel']ong to-us 13"),
                         "All your one point two bases are. SO bel ong to-us thirteen")

    def test_clear_corpus(self):
        self.assertListEqual(preprocessing.clear_corpus_characters(self.corpus_dirty, 1), self.corpus_clean)
