import os
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
from pathlib import Path
import concurrent.futures
from time import time

def read_pts_file(file_path):
    """
    Read a single .pts file and return its points.
    
    Parameters:
    -----------
    file_path : str
        Path to the .pts file
        
    Returns:
    --------
    tuple
        (file_name, points_array) or (file_name, None) if error
    """
    try:
        file_name = Path(file_path).stem
        points = np.loadtxt(file_path, delimiter=None)
        
        if points.shape[1] != 3:
            print(f"Warning: {file_name} has {points.shape[1]} columns instead of expected 3.")
            return file_name, None
            
        return file_name, points
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {str(e)}")
        return Path(file_path).stem, None

def visualize_multiple_segment_clouds_parallel(folder_path, file_extension='.pts', 
                                              max_workers=None, point_size=5.0, 
                                              background='black'):
    """
    Read multiple .pts files from a folder in parallel and visualize them as 
    point clouds with different colors for each segment.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing .pts files
    file_extension : str, optional
        Extension of point cloud files (default: '.pts')
    max_workers : int, optional
        Maximum number of threads to use (default: None, which uses the default
        ThreadPoolExecutor behavior - typically number of processors * 5)
    point_size : float, optional
        Size of the points in the visualization (default: 5.0)
    background : str or tuple, optional
        Background color of the visualization (default: 'black')
    
    Returns:
    --------
    plotter : pyvista.Plotter
        The PyVista plotter object
    clouds : dict
        Dictionary mapping file names to their corresponding point cloud data
    """
    # Get all .pts files in the directory
    pts_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.endswith(file_extension)]
    
    if not pts_files:
        raise ValueError(f"No {file_extension} files found in {folder_path}")
    
    print(f"Found {len(pts_files)} {file_extension} files. Starting parallel processing...")
    start_time = time()
    
    # Read all files in parallel
    segments = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file reading tasks
        future_to_file = {executor.submit(read_pts_file, file_path): file_path 
                         for file_path in pts_files}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_name, points = future.result()
                if points is not None:
                    segments[file_name] = points
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    end_time = time()
    print(f"Finished reading {len(segments)} point clouds in {end_time - start_time:.2f} seconds")
    
    # Now visualize the point clouds
    # Create a plotter
    plotter = pv.Plotter()
    plotter.background_color = background
    
    # Generate distinct colors for each segment
    color_list = list(mcolors.TABLEAU_COLORS.values())
    colors = [color_list[i % len(color_list)] for i in range(len(segments))]
    
    # Dictionary to store all point cloud objects
    clouds = {}
    
    # Process each segment and add to plotter
    for i, (file_name, points) in enumerate(segments.items()):
        # Create a PyVista point cloud
        cloud = pv.PolyData(points)
        
        # Add the point cloud to the plotter with its assigned color
        plotter.add_mesh(cloud, render_points_as_spheres=True, 
                         point_size=point_size, color=colors[i],
                         label=file_name)
        
        # Store the cloud in the dictionary
        clouds[file_name] = cloud
        
        print(f"Added '{file_name}' with {cloud.n_points} points")
    
    # Add coordinate axes for reference
    plotter.add_axes()
    
    # Add a legend to identify each segment
    if clouds:
        plotter.add_legend()
    
    # Show the point clouds
    plotter.show()
    
    return plotter, clouds

# Example usage
if __name__ == "__main__":
    # Replace with your folder path containing .pts files
    segment_folder = "/storage/student6/dev/pointnet.pytorch/training_dataset/train/"
    
    # You can specify the number of workers or let it use the default
    # For I/O bound tasks like file reading, using more workers than CPU cores can be beneficial
    plotter, clouds = visualize_multiple_segment_clouds_parallel(
        segment_folder, 
        max_workers=8  # Adjust based on your system
    )
    
    # Print summary of loaded point clouds
    print("\nSummary of loaded point clouds:")
    total_points = 0
    for name, cloud in clouds.items():
        print(f"Segment '{name}': {cloud.n_points} points")
        total_points += cloud.n_points
    print(f"Total points across all segments: {total_points}")