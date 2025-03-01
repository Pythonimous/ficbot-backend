import unittest
from scripts.data import download


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