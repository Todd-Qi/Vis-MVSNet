import os

# data_folder = "/mnt/A/xxx/2021/Github/Vis-MVSNet/eth3d/num_src7/test/"

# scenes = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office",
#           "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]
# scenes = ["botanical_garden", "boulders", "bridge", "door", "exhibition_hall", "lecture_room",
#           "living_room", "lounge", "observatory", "old_computer", "statue", "terrace_2"]

data_folder = "/mnt/A/qiyh/xxx/Github/Vis-MVSNet/tt/test/"

scenes = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]
scenes = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]

if __name__ == "__main__":

    point_folder = os.path.join(data_folder, "point_clouds")
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)

    for scene in scenes:
        source_ply = os.path.join(data_folder, scene, "final3d_model.ply")
        target_ply = os.path.join(point_folder, "{}.ply".format(scene))

        cmd = "mv " + source_ply + " " + target_ply
        print(cmd)
        os.system(cmd)
  





