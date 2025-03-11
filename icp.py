
import numpy as np
from scipy.spatial import KDTree
import time

def tic():
  return time.time()

def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def associate(sourceM, targetM):
    '''
    dim = xyz = 3 or xy=2
    source: np.array, shape (n, dim)
    target: np.array, shape (m, dim)
    '''
    dim = sourceM.shape[1]
    assert sourceM.shape[1] == dim
    assert targetM.shape[1] == dim
    
    source = sourceM.copy()
    target = targetM.copy()
    
    flipped = False
    if source.shape[0] < target.shape[0]:
        flipped = True
        source, target = target, source
    
    assert source.shape[0] >= target.shape[0]
    

    # Build a KDTree from source
    tree = KDTree(source)

    # Query the tree for nearest neighbors for each point in target
    distances, indices = tree.query(target)

    # Collect closest points in source for each point in target
    z = source[indices]
    #toc(t, "Associate tree")
    if flipped: z, target = target, z
    
    return z, target

def icp(source_pc_raw, target_pc_raw,R_0 = None,p_0 = None, samples = 1000,max_iter=1000,lossData = False):
    ''' 
    source_pc: 3D numpy array of shape (N, dim)
    target_pc: 3D numpy array of shape (M, dim)
    p_0: 3D numpy array of shape (dim, 1)
    R_0: 3D numpy array of shape (dim, dim)
    '''
    dim = source_pc_raw.shape[1]
    assert source_pc_raw.shape[1] == dim 
    assert target_pc_raw.shape[1] == dim

    if R_0 is None:
        R_0 = np.eye(dim)
    else:
        assert R_0.shape == (dim,dim)

    if p_0 is None:
        p_0 = np.mean(target_pc_raw, axis=0).reshape(dim,1) - np.mean(source_pc_raw, axis=0).reshape(dim,1)
    else:
        assert p_0.shape == (dim,1)

    # downsample the point clouds
    if source_pc_raw.shape[0] > samples:
        source_pc = source_pc_raw[np.random.choice(source_pc_raw.shape[0], samples, replace=False)]
    else:
        source_pc = source_pc_raw
    if target_pc_raw.shape[0] > samples:
        target_pc = target_pc_raw[np.random.choice(target_pc_raw.shape[0], samples, replace=False)]
    else:
        target_pc = target_pc_raw
    
    
    prev_loss = np.inf
    for i in range (max_iter):
        # transform source_pc
        new_source_pc = (R_0 @ source_pc.T + p_0).T
        
        z, m = associate(new_source_pc, target_pc) 
        loss = np.mean(np.linalg.norm(z - m, axis=1))
       
        if abs(prev_loss- loss) < 1e-6:
            if lossData:
                print('Converged at iteration: %d with loss of: %f' % (i-1, loss))
            break
        prev_loss = loss

        # revert transformation
        z = (np.linalg.inv(R_0)@(z.T-p_0)).T
        z_mean = np.mean(z, axis=0).reshape(dim,1)
        m_mean = np.mean(m, axis=0).reshape(dim,1)
        z_centered = z.T - z_mean
        m_centered = m.T - m_mean
        
        Q = m_centered @ z_centered.T
        U, S, V = np.linalg.svd(Q)
        A = np.eye(U.shape[1])
        A[-1,-1] = np.linalg.det(U@V)
        R_0 = U @ A @ V
        
        p_0= m_mean - R_0 @ z_mean
        
    
    T = np.eye(4)
    T[:dim, :dim] = R_0
    T[:dim, 3] = p_0.reshape(dim,)

    return T


if __name__ == "__main__":
  from utils import read_canonical_model, load_pc, visualize_icp_result
 
  num_pc = 4 # number of point clouds
for obj_name in ['drill', 'liq_container']:
    source_pc = read_canonical_model(obj_name)
    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)
        print("source_pc shape: ", source_pc.shape)
        print("target_pc shape: ", target_pc.shape)
        t = tic()
        # estimated_pose, you need to estimate the pose with ICP
        pose = icp(source_pc, target_pc)
        toc(t, "ICP: "+str(i) + " for " + obj_name)

        # visualize the estimated result
        visualize_icp_result(source_pc, target_pc, pose)

