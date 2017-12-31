import logging
import datetime


def initialize_logger():
    curr_date = datetime.date.today().strftime('%Y-%m-%d')
    log_filename = './logFiles/logfile_' + curr_date + '.log'
    log_format = '%(asctime)s %(message)s'
    log_dateformat = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt=log_dateformat,
                        filename=log_filename
                        )


def log(line):
    print(line)
    logging.debug(line)
