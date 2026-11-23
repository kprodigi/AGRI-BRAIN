import unittest
from pathlib import Path
import argparse
import tempfile
import study

class DesignTests(unittest.TestCase):
    def test_complete_crossing_and_counts(self):
        settings=study.design();tasks=study.tasks()
        self.assertEqual(len(settings),23);self.assertEqual(len(tasks),2300)
        self.assertEqual(len({(t['setting'],t['seed'],t['scenario']) for t in tasks}),2300)
        for setting in settings:
            self.assertEqual(sum(t['setting']==setting['id'] for t in tasks),100)

    def test_oat_is_single_factor_and_social_simplex(self):
        for s in study.design():
            p=s['effective_policy']
            self.assertAlmostEqual(sum(p[k] for k in ['w_c','w_l','w_r','w_p']),1,places=14)
            self.assertTrue(all(p[k]>0 for k in p))
            if s['family']=='weights' and s['id']!='nominal' and not s['id'].startswith('joint'):
                self.assertEqual(sum(v!=1 for v in s['factors'].values()),1)
        nominal=study.design()[0]['effective_policy']
        self.assertEqual([nominal[k] for k in ['w_c','w_l','w_r','w_p']],study.DEFAULT_SOCIAL)

    def test_invalid_factors_fail_closed(self):
        f=dict.fromkeys(study.GROUPS,1.0);f['eta']=float('nan')
        with self.assertRaises(ValueError):study.effective(f)
        with self.assertRaises(ValueError):study.effective({})

    def test_source_integrity(self):
        self.assertEqual(len(study.verify_source()),64)

    def test_live_overrides_and_restoration_after_exception(self):
        from mvp.simulation import generate_results as gr
        from src.models import action_selection
        from pirag import context_to_logits as ctx
        import numpy as np
        original=(gr.Policy,gr.SCENARIOS,gr.RESULTS_DIR,action_selection.THETA.copy(),ctx.THETA_CONTEXT.copy())
        setting=next(s for s in study.design() if s['id']=='joint20_A')
        with self.assertRaisesRegex(RuntimeError,'test restoration'):
            with study.overridden(gr,setting,'heatwave',Path(tempfile.gettempdir())) as receipt:
                policy=gr.Policy()
                self.assertEqual(policy.eta,setting['effective_policy']['eta'])
                self.assertAlmostEqual(policy.w_c,setting['effective_policy']['w_c'])
                np.testing.assert_array_equal(action_selection.THETA,original[3]*1.2)
                np.testing.assert_array_equal(ctx.THETA_CONTEXT[:,[0,1,4]],original[4][:,[0,1,4]]*.8)
                np.testing.assert_array_equal(ctx.THETA_CONTEXT[:,[2,3]],original[4][:,[2,3]]*1.2)
                raise RuntimeError('test restoration')
        self.assertIs(gr.Policy,original[0]);self.assertIs(gr.SCENARIOS,original[1])
        np.testing.assert_array_equal(action_selection.THETA,original[3]);np.testing.assert_array_equal(ctx.THETA_CONTEXT,original[4])

    def test_capability_contract(self):
        from src.models.mode_capabilities import capabilities_for
        no=capabilities_for('no_context');full=capabilities_for('agribrain')
        self.assertFalse(no.peer_messages);self.assertIsNone(no.context_kind)
        self.assertFalse(no.context_matrix_learning);self.assertTrue(no.policy_delta_learning)
        self.assertEqual(full.retrieval_kind,'standard')
        self.assertEqual(sum(capabilities_for(m).episode_count for m in study.MODES),9)

    def test_primary_reference_complete(self):
        refs=study.load(study.HERE/'primary_reference.json')['values']
        self.assertEqual(set(refs),set(map(str,study.SEEDS)))
        for seed in refs:
            self.assertEqual(set(refs[seed]),set(study.SCENARIOS))
            for modes in refs[seed].values():
                for r in modes.values(): self.assertEqual(len(r['action_trace']),288)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development-runtime',type=Path)
    args=p.parse_args();study.configure(args.development_runtime)
    unittest.main(argv=['test_study'],verbosity=2)
