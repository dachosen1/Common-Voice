
if __name__ == '__main__':
    from multiprocessing import Pool

    from audio_model.config.config import DataDirectory

    from audio_model.preprocessing.mp3_parser import Mp3parser
    from utlis import run_thread_pool, run_process_pool
    from audio_model.config.config import CommonVoiceModels

    import time

    parser = Mp3parser(
        data_path=DataDirectory.DATA_DIR,
        clips_dir=DataDirectory.CLIPS_DIR,
        document_path=DataDirectory.DEV_DIR,
        data_label='accent',
        model=CommonVoiceModels.Country,
    )

    mp3_list = range(0, 500)

    start = time.time()
    run_thread_pool(function=parser.convert_to_wav, my_iter=mp3_list)
    print('Took', int(time.time() - start), 'seconds.')

    # start = time.time()
    # for i in mp3_list:
    #     parser.convert_to_wav(i)
    # print('Took', int(time.time() - start), 'seconds.')

    p = Pool()
    start = time.time()
    p.map(parser.convert_to_wav, mp3_list)
    print('Took', int(time.time() - start), 'seconds.')
