"""
Registration engines: ANTs (antspyx) and SimpleITK, wrapped to a common
return format so the app can display and compare them side by side.

ANTs is also the only engine used for group template construction --
SimpleITK has no equivalent to ants.build_template().
"""
import time

import ants
import SimpleITK as sitk


def register_ants(fixed_path: str, moving_path: str, transform_type: str = "Rigid") -> dict:
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    start = time.time()
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type)
    elapsed = time.time() - start

    warped = reg["warpedmovout"]
    return {
        "engine": "ANTs",
        "elapsed_sec": elapsed,
        "fixed_array": fixed.numpy(),
        "moving_array": moving.numpy(),
        "warped_array": warped.numpy(),
        "fwdtransforms": reg["fwdtransforms"],
    }


def register_sitk(fixed_path: str, moving_path: str) -> dict:
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg_method = sitk.ImageRegistrationMethod()
    reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_method.SetMetricSamplingStrategy(reg_method.RANDOM)
    reg_method.SetMetricSamplingPercentage(0.2)
    reg_method.SetInterpolator(sitk.sitkLinear)
    reg_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200
    )
    reg_method.SetOptimizerScalesFromPhysicalShift()
    reg_method.SetInitialTransform(initial_tx, inPlace=False)
    reg_method.SetShrinkFactorsPerLevel([4, 2, 1])
    reg_method.SetSmoothingSigmasPerLevel([2, 1, 0])

    start = time.time()
    final_tx = reg_method.Execute(fixed, moving)
    elapsed = time.time() - start
    metric = reg_method.GetMetricValue()

    warped = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0)

    return {
        "engine": "SimpleITK",
        "elapsed_sec": elapsed,
        "metric": metric,
        "fixed_array": sitk.GetArrayFromImage(fixed),
        "moving_array": sitk.GetArrayFromImage(moving),
        "warped_array": sitk.GetArrayFromImage(warped),
    }


def build_group_template(t1_paths: list, iterations: int = 2):
    """
    Unbiased iterative template construction (ANTs only).
    Returns an ants.ANTsImage. Slow -- SyN registration runs
    `iterations` times across all subjects.
    """
    images = [ants.image_read(p) for p in t1_paths]
    template = ants.build_template(
        image_list=images,
        type_of_transform="SyN",
        iterations=iterations,
    )
    return template