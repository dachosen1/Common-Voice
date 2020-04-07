def get_batches(*, arr, batch_size, seq_length):
    """

    """

    n_batches = len(arr) // (batch_size * seq_length)
    arr = arr[: n_batches * (batch_size * seq_length)].reshape((batch_size, -1))

    for i in range(0, arr.shape[1]):
        x = arr[:, i : i + seq_length]
        y = arr[:, i + seq_length : i + seq_length + 1]

        if x.shape[1] != seq_length:
            break

        yield x, y
