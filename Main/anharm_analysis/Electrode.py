import pandas as pd
import numpy as np 
from .Grid import COMSOLGrid

class SimulatedElectrode: 
    def __init__(self, name, file, sim_unit=1e-6): 
        self.V, self.Ex, self.Ey, self.Ez, self.sim_grid = [], [], [], [], [] 

    def load_from_file(self): 
        raise NotImplementedError 

    def get_V_in_cube(self, V0=1, L_cube=None): 
        if L_cube is None: 
            return V0 * self.V 
        x0, y0, z0 = self.sim_grid.get_grid_center() 
        V_idx = self.sim_grid.get_subgrid_xyzi(x0-L_cube/2, x0+L_cube/2, 
                                               y0-L_cube/2, y0+L_cube/2, 
                                               z0-L_cube/2, z0+L_cube/2)[-1] 
        return V0 * self.V[V_idx]

    def get_subcube_Vxyz(self, V0=1, L_cube=None): 
        if L_cube is None: 
            return V0 * self.V 
        x0, y0, z0 = self.sim_grid.get_grid_center() 
        X, Y, Z, V_idx = self.sim_grid.get_subgrid_xyzi(x0-L_cube/2, x0+L_cube/2, 
                                               y0-L_cube/2, y0+L_cube/2, 
                                               z0-L_cube/2, z0+L_cube/2)
        return V0 * self.V[V_idx], X, Y, Z

    def set_sim_grid(self, grid): 
        self.sim_grid = grid
    
    def get_V_at_points(self, x, y, z, V0=1, indices=None):
        """Get potential values at specific points (x, y, z) by finding matching points in sim_grid.
        
        If indices are provided (e.g., from ROI_grid.indices), use them directly for efficiency.
        Otherwise, find indices by matching coordinates.
        """
        if indices is not None:
            # Use provided indices directly (most efficient)
            return V0 * self.V[indices]
        
        # Fallback: find indices by coordinate matching
        # Since ROI_grid points come from sim_grid, we can find indices more efficiently
        # by using the boundaries to get the subgrid indices
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # Get all points in this region from sim_grid
        _, _, _, region_indices = self.sim_grid.get_subgrid_xyzi(x_min, x_max, y_min, y_max, z_min, z_max)
        
        # Now match the specific points (ROI_grid points should be a subset of these)
        # Create a mapping using coordinate matching
        tol = 1e-10
        sim_x = self.sim_grid.x[region_indices]
        sim_y = self.sim_grid.y[region_indices]
        sim_z = self.sim_grid.z[region_indices]
        
        # Find matching indices
        matched_indices = []
        for xi, yi, zi in zip(x, y, z):
            mask = (np.abs(sim_x - xi) < tol) & \
                   (np.abs(sim_y - yi) < tol) & \
                   (np.abs(sim_z - zi) < tol)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                matched_indices.append(region_indices[idx[0]])
            else:
                raise ValueError(f"Point ({xi}, {yi}, {zi}) not found in sim_grid")
        
        return V0 * self.V[np.array(matched_indices)]

    
class COMSOLElectrode(SimulatedElectrode): 
    def __init__(self, name, file, sim_unit=1e-6, 
                 header_x='% x', header_y='y', header_z='z', **kwargs): 
        self.V, self.Ex, self.Ey, self.Ez, self.sim_grid = self.load_from_file(name, file, 
                                                                               unit=sim_unit, 
                                                                               header_x=header_x, 
                                                                               header_y=header_y, 
                                                                               header_z=header_z, **kwargs) 

    def load_from_file(self, electrode, file, excitation_prefix='V', sim_prefix='esbe.', skiprows=8, 
                       unit=1e-6, header_x='% x', header_y='y', header_z='z', **kwargs): 
        df = pd.read_csv(file, skiprows=skiprows) 
        all_headers = [i for i in df.keys() if f'{excitation_prefix}{electrode}=1' in i] 
        V_headers = [i for i in all_headers if f'{sim_prefix}V' in i] 
        Ex_headers = [i for i in all_headers if f'{sim_prefix}Ex' in i]
        Ey_headers = [i for i in all_headers if f'{sim_prefix}Ey' in i] 
        Ez_headers = [i for i in all_headers if f'{sim_prefix}Ez' in i] 
        assert len(V_headers)==1, "Check output results or output headers, found 0 or 1+ potential data"
        V = np.array(df[V_headers[0]])
        Ex = np.array([]) if len(Ex_headers) != 1 else np.array(df[Ex_headers[0]]) 
        Ey = np.array([]) if len(Ey_headers) != 1 else np.array(df[Ey_headers[0]]) 
        Ez = np.array([]) if len(Ez_headers) != 1 else np.array(df[Ez_headers[0]]) 
        grid = COMSOLGrid(df, unit, header_x, header_y, header_z) 
        return V, Ex, Ey, Ez, grid