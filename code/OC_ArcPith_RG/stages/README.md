# Grayscale data preparation and stage runner

Run from the package directory with the bundled Python environment:

```powershell
$root = "D:\教务处实习\wood_preproject\树髓定位\代码\OC_ArcPith_RG_72h_algorithm_package_adjusted 2\OC_ArcPith_RG_72h_algorithm_package_adjusted"
$out = "D:\教务处实习\wood_preproject\树髓定位\代码\最终产物"
python "$root\tools\prepare_grayscale_manifest.py" --manifest "$root\data\manifest.jsonl" --output-root "$root\data"
python "$root\stages\stage0_contract.py" --manifest "$root\data\manifest_grayscale.jsonl" --output-dir "$out\stage 0 grayscale"
python "$root\stages\stage1_continuous.py" --manifest "$out\stage 0 grayscale\manifest_validated.jsonl" --output-dir "$out\stage 1 grayscale"
python "$root\stages\stage2_preflight_coldstart.py" --stage1-dir "$out\stage 1 grayscale" --output-dir "$out\stage 2 grayscale"
python "$root\stages\stage3_all_arc.py" --stage1-dir "$out\stage 1 grayscale" --stage2-dir "$out\stage 2 grayscale" --output-dir "$out\stage 3 grayscale"
python "$root\stages\stage4_profiles.py" --stage1-dir "$out\stage 1 grayscale" --stage3-dir "$out\stage 3 grayscale" --output-dir "$out\stage 4 grayscale"
python "$root\stages\stage5_risk_execution.py" --stage0-dir "$out\stage 0 grayscale" --stage1-dir "$out\stage 1 grayscale" --stage3-dir "$out\stage 3 grayscale" --stage4-dir "$out\stage 4 grayscale" --output-dir "$out\stage 5 grayscale"
python "$root\stages\stage6_contributions.py" --stage1-dir "$out\stage 1 grayscale" --stage4-dir "$out\stage 4 grayscale" --output-dir "$out\stage 6 grayscale"
```

The derived manifest keeps the color crop in `image_color` and points `image`
to the dataset_grid-compatible grayscale ring raster. Stage 0 freezes the
contract, Stage 1 freezes continuous evidence, Stage 2 writes candidate-free
descriptors plus the six-family RP2 seed registry, and Stage 3 performs the
certified All-Arc inversion. Stage 4 performs high-resolution refinement and
projective profile classification. Stage 5 adds model-risk and execution
gates; Stage 6 performs contribution screening and budgeted exact
delete-refit. Stage 7 is not run by these commands.
