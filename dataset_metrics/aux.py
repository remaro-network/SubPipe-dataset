import numpy as np
from scipy.spatial.transform import Rotation as R

def print_header(title, width=70, char='*'):
    """
    Prints a formatted header with a title centered.

    Parameters:
    title (str): The title to display in the header.
    width (int): The total width of the header. Default is 70.
    char (str): The character used for the border. Default is '*'.
    """
    border = char * width
    formatted_title = title.center(width, char)
    
    print(border)
    print(formatted_title)
    print(border)

def tvector_2_se3(transforms):
    ''' 
    :param: T: Flattened Transform matrix
    :returns: translation vector t and so3 lie algebra vector
    '''
    t_vectors = []
    for T in transforms:
        flat_SE3 = T[0:12]
        se3s = SEs_2_ses(flat_SE3)
        t_vectors.append(se3s)
    return np.asarray(t_vectors).reshape(len(t_vectors),6)

def SEs_2_ses(motion_data):
    motion_data = np.array(motion_data).reshape(1,12)
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE_2_se(SE)
    return ses

def SE_2_se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO_2_so(SE_data[0:3,0:3]).T
    return result

def SO_2_so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()
