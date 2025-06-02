from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Image config
        DeclareLaunchArgument('image.width', default_value='1280'),
        DeclareLaunchArgument('image.height', default_value='720'),
        DeclareLaunchArgument('image.left_border', default_value='66'),
        # Robot config
        DeclareLaunchArgument('robot.grid_height', default_value='26'),
        DeclareLaunchArgument('robot.row', default_value='14'),
        DeclareLaunchArgument('robot.col', default_value='77'),
        # Grid config
        DeclareLaunchArgument('grid.position.x', default_value='34.0'),
        DeclareLaunchArgument('grid.position.y', default_value='73.0'),
        DeclareLaunchArgument('grid.position.z', default_value='0.0'),
        # Camera config
        DeclareLaunchArgument('camera.zed_height', default_value='64'),
        DeclareLaunchArgument('camera.zed_fov_vert', default_value='68.0'),
        DeclareLaunchArgument('camera.zed_fov_horz', default_value='101.0'),
        DeclareLaunchArgument('camera.zed_camera_tilt', default_value='65.0'),
        # Topics config 65
        DeclareLaunchArgument('topics.image_subscription', default_value='/zed/zed_node/left/image_rect_color'),
        DeclareLaunchArgument('topics.occupancy_grid', default_value='/test_occ'),
        # Sizes config
        DeclareLaunchArgument('sizes.desired_size', default_value='0.05'),
        DeclareLaunchArgument('sizes.curr_pix_size', default_value='0.0055'),
        # Offsets config
        DeclareLaunchArgument('offsets.polygon_offset_right', default_value='17'),
        DeclareLaunchArgument('offsets.polygon_offset_top', default_value='75'),
        # HSV filter config
        DeclareLaunchArgument('hsv.lower', default_value='[0, 0, 136]'),
        DeclareLaunchArgument('hsv.upper', default_value='[179, 36, 255]'),
        # Morphology config 
        DeclareLaunchArgument('morph.iterations', default_value='2'),

        Node(
            package='drivable_area',
            executable='drivable_area_self',
            name='drivable_area',
            output='screen',
            parameters=[{
                'image.width': LaunchConfiguration('image.width'),
                'image.height': LaunchConfiguration('image.height'),
                'image.left_border': LaunchConfiguration('image.left_border'),
                'robot.grid_height': LaunchConfiguration('robot.grid_height'),
                'robot.row': LaunchConfiguration('robot.row'),
                'robot.col': LaunchConfiguration('robot.col'),
                'grid.position.x': LaunchConfiguration('grid.position.x'),
                'grid.position.y': LaunchConfiguration('grid.position.y'),
                'grid.position.z': LaunchConfiguration('grid.position.z'),
                'camera.zed_height': LaunchConfiguration('camera.zed_height'),
                'camera.zed_fov_vert': LaunchConfiguration('camera.zed_fov_vert'),
                'camera.zed_fov_horz': LaunchConfiguration('camera.zed_fov_horz'),
                'camera.zed_camera_tilt': LaunchConfiguration('camera.zed_camera_tilt'),
                'topics.image_subscription': LaunchConfiguration('topics.image_subscription'),
                'topics.occupancy_grid': LaunchConfiguration('topics.occupancy_grid'),
                'sizes.desired_size': LaunchConfiguration('sizes.desired_size'),
                'sizes.curr_pix_size': LaunchConfiguration('sizes.curr_pix_size'),
                'offsets.polygon_offset_right': LaunchConfiguration('offsets.polygon_offset_right'),
                'offsets.polygon_offset_top': LaunchConfiguration('offsets.polygon_offset_top'),
                'hsv.lower': LaunchConfiguration('hsv.lower'),
                'hsv.upper': LaunchConfiguration('hsv.upper'),
                'morph.iterations': LaunchConfiguration('morph.iterations'),
            }]
        )
    ])
