# ASYMMETRIC DIFFERENTIAL LAYER
def Adl_2d(x):
    """
    Asymmetric Differential Layer (ADL) in 2D.
    
    Given a matrix IN of shape *[H, W]*, returns a matrix OUT of shape *[H , integer_part(W/2)]*.
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    
    Parameters
    -----------------
    x : torch.Tensor
        The input matrix *IN* (size *HxW*)
        
    Returns
    -----------------
    torch.Tensor 
        ADL output matrix *OUT* (size *Hx(W/2)*).
        OUT(i, j) = IN(i, j) - IN(i, W + 1 - j).
    """
    # Given a matri
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :w_2]
    dx = x[:, -w_2:]
    return sx - dx.flip(-1)

def Adl_3d(x):
    """
    Asymmetric Differential Layer (ADL) in 3D.
    
    Given a matrix IN of shape *[F, H, W]*, returns a matrix OUT of shape *[F, H , integer_part(W/2)]*.
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    ADL in 3D is equivalent to perform ADL in 2D F times, one for HxW matrix in input.
    
    Parameters
    -----------------
    x : torch.Tensor
        The input matrix *IN* (size *FxHxW*)
        
    Returns
    -----------------
    torch.Tensor 
        ADL output matrix *OUT* (size *FxHx(W/2)*).
        OUT(k, i, j) = IN(k, i, j) - IN(k, i, W + 1 - j).
    
    See also
    ----------------
    Adl_2d : Asymmetric Differential Layer in 2D
    """
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :, :w_2]
    dx = x[:, :, -w_2:]
    return sx - dx.flip(-1)

def Adl_4d(x):
    """
    Asymmetric Differential Layer (ADL) in 4D.
    
    Given a matrix IN of shape *[B, F, H, W]*, returns a matrix OUT of shape *[B, F, H , integer_part(W/2)]*.
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    ADL in 4D is equivalent to perform ADL in 3D B times, one for FxHxW matrix in input.
    
    Parameters
    -----------------
    x : torch.Tensor
        The input matrix *IN* (size *BxFxHxW*)
        
    Returns
    -----------------
    torch.Tensor 
        ADL output matrix *OUT* (size *BxFxHx(W/2)*).
        OUT(b, k, i, j) = IN(b, k, i, j) - IN(b, k, i, W + 1 - j).
        
    See also
    ----------------
    Adl_3d : Asymmetric Differential Layer in 2D
    """
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :, :, :w_2]
    dx = x[:, :, :, -w_2:]
    return sx - dx.flip(-1)