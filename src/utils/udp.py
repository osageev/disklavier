import socket

def send_udp(message: str, address: str = "/midi", ip: str = "127.0.0.1", port: int = 7400):
    """
    send a message via udp as an osc packet

    Parameters
    ----------
    message : str
            message to send via udp
    ip : str
            ip address to send to
    port : int
            port to use

    Returns
    -------
    None
    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # format address with null padding to multiple of 4
    address_padded = address.encode("utf-8") + b"\0" * (4 - (len(address) % 4))
    if len(address) % 4 == 0:
        address_padded += b"\0" * 4

    # format type tags (s for string)
    type_tag = ",s"
    type_tag_padded = type_tag.encode("utf-8") + b"\0" * (4 - (len(type_tag) % 4))
    if len(type_tag) % 4 == 0:
        type_tag_padded += b"\0" * 4

    # format string argument with null padding
    arg_encoded = message.encode("utf-8")
    padding = 4 - (len(arg_encoded) % 4)
    if padding == 4:
        padding = 0
    arg_padded = arg_encoded + b"\0" * padding

    # combine all parts
    packet = address_padded + type_tag_padded + arg_padded

    udp_socket.sendto(packet, (ip, port))
    udp_socket.close()
