import os


def get_speed_stats(result):

    sps_total = result["timesteps_total"] / result["time_total_s"]
    sps_iter = result["timesteps_this_iter"] / result["time_this_iter_s"]
    sps_trained = result["info"]["num_steps_trained"] / result["time_total_s"]
    sps_sampled = result["info"]["num_steps_sampled"] / result["time_total_s"]

    tt1m_total = 1e6 / (sps_total * 3600.0) if sps_total > 0 else -1
    tt1m_trained = 1e6 / (sps_trained * 3600.0) if sps_trained > 0 else -1

    return sps_total, sps_iter, sps_trained, sps_sampled, tt1m_total, tt1m_trained


def profile_ray(curr_iter, debug_dir, ray):
    # profiling
    print("\n>>> Ray profiling/debugging info (WARNING: LONG)")
    print("Ray nodes:", ray.nodes())
    # print("Ray tasks:", ray.tasks())
    # print("Ray objects:", ray.objects())
    ray.timeline(
        filename=os.path.join(debug_dir, "ray-timeline-iter-{}.json".format(curr_iter))
    )
    ray.object_transfer_timeline(
        filename=os.path.join(
            debug_dir, "ray-object-transfer-timeline-iter-{}.json".format(curr_iter)
        )
    )
    print("Ray cluster_resources:", ray.cluster_resources())
    print("Ray available_resources:", ray.available_resources())
    print("Ray errors:", ray.errors())
    print("\n<<< End Ray profiling/debugging info\n\n")