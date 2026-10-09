import unittest,copy,math
import numpy as np
from oc_arcpith_rg.geometry import h_from_point,point_from_h,h_from_phi_kappa
from oc_arcpith_rg.candidates import build
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.io import Sample,Ring
from oc_arcpith_rg.objectives import objective,radial_pair_residuals
from oc_arcpith_rg.cv import fold_map
from oc_arcpith_rg.bias import audit
from oc_arcpith_rg.observability import support_components,classify,_m0_profile_grid
from oc_arcpith_rg.io import sample_from_dict
from oc_arcpith_rg.evidence import split_parent_rings,fit_objective,independent_audit
from oc_arcpith_rg.preflight import preflight
from oc_arcpith_rg.baselines import b0_one_ring_pseudocenters,b1_objective,b3_dense_search,compare_baselines,_grid_candidates

def cfg():
 return {'seed':7,'preprocess':{'resample_spacing_px':4.0,'spline_smoothing_px':0.1,'micro_arc_points':20,'ring_budget_max':1.0,'ring_budget_tau_norm':0.15,'min_points_per_ring':8},'model':{'student_t_nu':4.0,'sigma_abs':0.08,'m1_shape_sigma':0.1,'m1_order_sigma':0.04,'m1_lambda_shape':0.3,'m1_lambda_shape_grid':[0.1,0.3],'m1_lambda_order':0.08,'m1_axis_taper_kappa':0.03,'radial_profile_bins':32,'radial_profile_min_overlap_deg':10.0,'optimizer_maxiter':50,'random_starts':2,'ransac_trials':4,'ransac_keep':2,'mode_loss_delta':0.5,'mode_separation_norm':0.08},'candidate_screen':{'translation_fractions':[0.01,0.05],'direction_deg':[5,15],'distance_scales':[0.5,2.0],'random_candidates':3,'local_loss_tolerance':0.02,'global_truth_beats_min':0.8,'direction_tree_pass_fraction':0.7,'range_tree_pass_fraction':0.6,'local_bias_tree_pass_fraction':0.7},'bias_audit':{'radius_fraction':0.1,'coarse_grid':7,'gt_tolerance_fraction':0.01,'repair_min_bias_reduction':0.2,'repair_min_tree_fraction':0.7},'profile':{'phi_bins':12,'kappa_bins':8,'kappa_max':50.0,'support_delta':0.75,'point_phi_width_deg':15,'axis_phi_width_deg':20},'cv':{'folds':2,'finite_true_distance_max_norm':2.5,'min_median_point_improvement_norm':0.02,'min_improved_folds':1,'max_far_axis_degradation_deg':2.0},'contribution':{'max_groups':24,'ridge':1e-6}}

def circle(c,r,n=120,span=(-1.2,1.2)):
 th=np.linspace(*span,n);return np.column_stack([c[0]+r*np.cos(th),c[1]+r*np.sin(th)])
def sample(size=512):
 c=np.array([210.,260.]);rings=[Ring(str(i),circle(c,r),i) for i,r in enumerate([70,100,130,160])];return Sample('S','T',(size,size),rings,c)
class Core(unittest.TestCase):
 def test_h_roundtrip(self):
  p=np.array([3.2,-1.5]);self.assertLess(np.linalg.norm(point_from_h(h_from_point(p))-p),1e-10)
 def test_axis(self):self.assertIsNone(point_from_h(h_from_phi_kappa(0.3,0.0)))
 def test_candidate_normalized_offsets(self):
  c=cfg();p1=prepare(sample(512),c);lib=build(p1,c);offs=sorted({x['offset_norm'] for x in lib if x['kind']=='translation'});self.assertEqual(offs,[0.01,0.05])
 def test_candidate_library_deterministic(self):
  c=cfg();p=prepare(sample(),c);a=[x['name'] for x in build(p,c)];b=[x['name'] for x in build(p,c)];self.assertEqual(a,b)
 def test_m0_circle_prefers_gt_to_large_shift(self):
  c=cfg();p=prepare(sample(),c);gt=h_from_point((p.sample.pith_px-p.center_px)/p.scale_px);shift=h_from_point((p.sample.pith_px-p.center_px)/p.scale_px+np.array([0.2,0]));self.assertLess(objective(gt,p,c,'m0'),objective(shift,p,c,'m0'))
 def test_rpc_correct_center_small_shape_residual(self):
  c=cfg();p=prepare(sample(),c);gt=h_from_point((p.sample.pith_px-p.center_px)/p.scale_px);sh,od=radial_pair_residuals(gt,p.points_by_ring['0'][0],p.points_by_ring['1'][0],c);self.assertLess(float(np.median(np.abs(sh))),0.05);self.assertLess(float(np.max(od)),0.05)
 def test_multiple_fragments_are_retained_without_duplicate_order_or_arc_ids(self):
  c=cfg();s=sample();s.rings.append(Ring('1',circle(np.array([210.,260.]),100,span=(1.5,2.2)),1));p=prepare(s,c)
  self.assertEqual(len(p.points_by_ring['1']),2);self.assertEqual(p.ring_order,['0','1','2','3']);self.assertEqual(len({a.arc_id for a in p.arcs}),len(p.arcs))
 def test_missing_parent_ring_is_not_made_adjacent(self):
  c=cfg();s=sample();s.rings=[s.rings[0],s.rings[2],s.rings[3]];s.metadata={'ring_order_audit':{'full_parent_ring_order':['0','1','2','3']}};p=prepare(s,c)
  self.assertEqual(p.ring_order,['0','1','2','3'])
 def test_equal_parent_total_budget_is_independent_of_visible_arc_length(self):
  c=cfg(); center=np.array([210.,260.])
  # Keep all rings eligible while deliberately giving one parent ring a much
  # shorter visible arc. Equal-parent means its total omega must not be
  # reduced merely because less of that ring is visible.
  rings=[Ring('0',circle(center,70,n=24,span=(-0.15,0.15)),0),
         Ring('1',circle(center,100,n=120,span=(-1.2,1.2)),1),
         Ring('2',circle(center,130,n=120,span=(-1.2,1.2)),2)]
  s=Sample('equal-parent','T',(512,512),rings,center)
  p=prepare(s,c)
  totals={}
  for a in p.arcs: totals[a.ring_id]=totals.get(a.ring_id,0.0)+a.omega
  self.assertEqual(set(totals),{'0','1','2'})
  self.assertTrue(np.allclose(list(totals.values()),c['preprocess']['ring_budget_max'],atol=1e-12,rtol=0),totals)
 def test_bias_audit_cannot_escape_preregistered_radius(self):
  c=cfg();c['bias_audit']['radius_fraction']=0.05;c['bias_audit']['coarse_grid']=5;p=prepare(sample(),c);r=audit(p,c,'m0')
  self.assertLessEqual(r['bias_norm'],0.050001)
 def test_observability_components_wrap_phi(self):
  support=np.zeros((3,5),bool);support[1,0]=True;support[1,-1]=True;support[2,2]=True
  self.assertEqual(support_components(support),[2,1])
 def test_observability_multimodal_requires_multiple_components(self):
  c=cfg();self.assertEqual(classify({'support_component_count':2,'includes_axis':False,'phi_width_deg':5},c),'MULTIMODAL')
  self.assertEqual(classify({'support_component_count':1,'includes_axis':False,'phi_width_deg':30},c),'REJECT')
 def test_observability_has_provisional_range_state(self):
  c=cfg();c['profile']['range_kappa_ratio']=2.0;c['profile']['range_phi_width_deg']=45.0
  prof={'support_component_count':1,'includes_axis':False,'phi_width_deg':20.0,
        'support_kappa_min':1.0,'support_kappa_max':4.0}
  self.assertEqual(classify(prof,c),'RANGE_UNCERTAIN')
 def test_candidate_free_preflight_does_not_require_candidate_loss(self):
  c=cfg();p=prepare(sample(),c);r=preflight(p,c,'P0')
  self.assertEqual(r['route'],'POINT_CANDIDATE')
  self.assertIn('eligible_ring_count',r['features'])
  self.assertNotIn('loss',r['features'])
  def test_preflight_blocks_point_when_audit_is_underpowered(self):
   c=cfg();s=sample();s.rings=s.rings[:2];p=prepare(s,c);r=preflight(p,c,'P0')
   self.assertEqual(r['route'],'REJECT');self.assertFalse(r['features']['audit_feasible'])
 def test_evidence_split_and_audit_firewall(self):
  c=cfg();p=prepare(sample(),c);part=split_parent_rings(p,c)
  self.assertTrue(part.fit_feasible);self.assertTrue(part.audit_feasible)
  h=h_from_point((p.sample.pith_px-p.center_px)/p.scale_px)
  before=fit_objective(h,p,c,part)
  audit1=independent_audit(h,p,c,part,'m0',[])
  # Changing the audit hypothesis set cannot alter the already-fitted score.
  audit2=independent_audit(h,p,c,part,'m0',[h_from_point([2.0,2.0])])
  self.assertEqual(before,fit_objective(h,p,c,part))
  self.assertEqual(audit1['audit_ring_count'],audit2['audit_ring_count'])
  self.assertEqual(audit1['outcome'],'HOLDOUT_UNDERPOWERED')
 def test_ingestion_attaches_replay_hashes(self):
  d={'sample_id':'hashes','tree_id':'T','section_id':'S','crop_id':'C',
     'image_size':[32,32],'curves':[{'ring_id':'0','fragment_id':'0:0','order':0,
     'points_px':[[1,1],[2,2],[3,3],[4,4]]}]}
  a=sample_from_dict(d);b=sample_from_dict(d)
  for key in ('input_hash','observation_hash','lineage_hash'):
   self.assertRegex(a.metadata[key],r'^[0-9a-f]{64}$');self.assertEqual(a.metadata[key],b.metadata[key])
  with self.assertRaises(ValueError):
   sample_from_dict({**d,'metadata':{'input_hash':'spoof'}})
 def test_stage1_baselines_are_defined_and_finite(self):
  c=cfg();p=prepare(sample(),c);h0=b0_one_ring_pseudocenters(p)
  self.assertIsNotNone(h0);self.assertTrue(np.isfinite(b1_objective(h0,p,c)))
  h3,l3=b3_dense_search(p,c,phi_bins=12,kappa_bins=8)
  self.assertIsNotNone(h3);self.assertTrue(np.isfinite(l3))
  rows=compare_baselines(p,c,h_b2=h0,b3_phi_bins=12,b3_kappa_bins=8,b1_phi_bins=12,b1_kappa_bins=8)
  self.assertEqual(set(rows),{'B0','B1','B2','B3'})
 def test_vectorized_b1_grid_matches_scalar_objective(self):
  c=cfg();p=prepare(sample(),c);loss,h=_grid_candidates(p,c,'b1',7,4,3.0)
  phis=np.linspace(-np.pi,np.pi,7,endpoint=False);kappas=np.r_[0.0,np.geomspace(1e-3,3.0,3)]
  scalar=min(b1_objective(h_from_phi_kappa(phi,kappa),p,c) for kappa in kappas for phi in phis)
  self.assertAlmostEqual(loss,scalar,places=12)
 def test_vectorized_m0_profile_matches_scalar_objective(self):
  c=cfg();p=prepare(sample(),c);phis=np.linspace(-np.pi,np.pi,7,endpoint=False);kappas=np.array([0.0,0.01,0.2,2.0])
  fast=_m0_profile_grid(p,c,phis,kappas,batch_size=3);slow=np.array([[objective(h_from_phi_kappa(phi,kappa),p,c,'m0') for phi in phis] for kappa in kappas])
  self.assertTrue(np.allclose(fast,slow,atol=1e-12,rtol=1e-12),np.max(np.abs(fast-slow)))
 def test_tree_folds_grouped(self):
  m=fold_map(['A','A','B','C','C'],2,3);self.assertEqual(m['A'],m['A']);self.assertEqual(m['C'],m['C'])
if __name__=='__main__':unittest.main()
