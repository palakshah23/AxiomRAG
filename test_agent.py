"""Quick regression tests for critic parsing; no Ollama call required."""
import unittest
import agent

class CriticTests(unittest.TestCase):
    def test_yes_yes(self):
        self.assertEqual(agent._parse_critic("GROUNDED: YES\nRELEVANT: YES"),(True,True,""))
    def test_no_grounding(self):
        g,r,_=agent._parse_critic("GROUNDED: NO\nRELEVANT: YES")
        self.assertFalse(g); self.assertTrue(r)
    def test_malformed_fails_closed(self):
        g,r,reason=agent._parse_critic("probably yes")
        self.assertFalse(g); self.assertFalse(r); self.assertTrue(reason)

if __name__=="__main__": unittest.main(verbosity=2)
