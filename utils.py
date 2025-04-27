import io
import struct

def create_vorbis_comment_block(comment_dict, last_block=True):
    """
    Create a vorbis comment block for FLAC metadata
    
    Args:
        comment_dict: Dictionary of comments to add
        last_block: Whether this is the last metadata block
        
    Returns:
        bytes: The formatted vorbis comment block
    """
    vendor_string = b'ComfyUI-MediaURLLoader'
    vendor_length = len(vendor_string)
    
    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)
    
    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)
    
    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    
    if last_block:
        id = b'\x84'  # Type 4 (VORBIS_COMMENT) with last-block flag set
    else:
        id = b'\x04'  # Type 4 (VORBIS_COMMENT)
    
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data
    
    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    """
    Insert or replace vorbis comments in a FLAC file
    
    Args:
        flac_io: BytesIO containing FLAC data
        comment_dict: Dictionary of comments to add
        
    Returns:
        BytesIO: New FLAC data with updated comments
    """
    if len(comment_dict) == 0:
        return flac_io
    
    flac_io.seek(4)  # Skip "fLaC" marker
    
    blocks = []
    last_block = False
    
    # Read existing blocks
    while not last_block:
        header = flac_io.read(4)
        if not header or len(header) < 4:
            break
            
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        
        block_data = flac_io.read(block_length)
        
        # Skip existing vorbis comment blocks and padding blocks
        if block_type == 4 or block_type == 1:
            pass
        else:
            # Clear the last-block flag
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)
    
    # Add our vorbis comment block (with last-block flag set)
    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))
    
    # Create new FLAC file
    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')  # FLAC signature
    
    for block in blocks:
        new_flac_io.write(block)
    
    # Add any remaining data
    new_flac_io.write(flac_io.read())
    
    # Reset position for reading
    new_flac_io.seek(0)
    
    return new_flac_io

def create_timestamp():
    """
    Create a timestamp string for unique filenames
    
    Returns:
        str: A timestamp string
    """
    import time
    import hashlib
    
    # Use current time plus a random component
    timestamp = str(time.time())
    unique_id = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    return unique_id 