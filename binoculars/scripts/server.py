'''Serverqueue where jobs can be submitted. Jobs will be calculated on
the spot or passed on to the OAR cluster if so specified in the
configfile. Jobs can be submitted in a json dictionary. The keyword
'command' and 'configfilename' supply a string with the command and
the path to the configfile. Everything else is assumed to be an
override in the configfile. If an override cannot be parsed the job
will start anyway without the override. The processingqueue cannot be
interrupted.

'''
import socket
import threading
import time
import sys
import traceback
import json
import os

import socketserver
import queue

import binoculars.main
import binoculars.util


class ProcessTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        input = self.request.recv(1024)
        if input.startswith('test'):
            print('Recieved test request')
            self.request.sendall('Connection succesful')
        else:
            try:
                job = json.loads(input)
                parsed, result = parse_job(job)
                if parsed:
                    print('Recieved command: {}. Job is added to queue.\nNumber of jobs left in queue: {}'.format(job['command'], self.server.q.qsize()))
                    response = 'Job added to queue'
                    self.server.q.put(job)
                else:
                    response = result
            except Exception:
                print(f'Could not parse the job: {input}')
                print(traceback.format_exc())
                response = 'Error: Job could not be added to queue'
            finally:
                self.request.sendall(response)


def parse_job(job):
    try:
        overrides = []
        for key in list(job.keys()):
            if key not in ['command', 'configfilename']:
                section_key, value = job[key].split('=')
                section, key = section_key.split(':')
                overrides.append((section, key, value))
        return True, overrides
    except Exception:
        message = f'Error parsing the configuration options. {job}'
        return False, message


def process(run_event, ip, port, q):
    while run_event.is_set():
        if q.empty():
            time.sleep(1)
        else:
            job = q.get()
            # assume everything in the list is an override except for command and configfilename
            command = str(job['command'])
            configfilename = job['configfilename']
            overrides = parse_job(job)[1]  # [1] are the succesfully parsed jobs
            print(f'Start processing: {command}')
            try:
                configobj = binoculars.util.ConfigFile.fromtxtfile(configfilename, overrides=overrides)
                if binoculars.util.parse_bool(configobj.dispatcher['send_to_gui']):
                    configobj.dispatcher['host'] = ip
                    configobj.dispatcher['port'] = port
                binoculars.main.Main.from_object(configobj, [command])
                print(f'Succesfully finished processing: {command}.')
            except Exception:
                errorfilename = f'error_{command}.txt'
                print(f'An error occured for scan {command}. For more information see {errorfilename}')
                with open(errorfilename, 'w') as fp:
                    traceback.print_exc(file=fp)
            finally:
                print(f'Number of jobs left in queue: {q.qsize()}')

def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
        port = sys.argv[2]
    else:
        ip = None
        port = None

    q = queue.Queue()

    binoculars.util.register_python_executable(os.path.join(os.path.dirname(__file__), 'binoculars.py'))

    HOST, PORT = socket.gethostbyname(socket.gethostname()), 0

    run_event = threading.Event()
    run_event.set()

    process_thread = threading.Thread(target=process, args=(run_event, ip, port, q))
    process_thread.start()

    server = socketserver.TCPServer((HOST, PORT), ProcessTCPHandler)
    server.q = q
    ip, port = server.server_address

    print(f'Process server started running at ip {ip} and port {port}. Interrupt server with Ctrl-C')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        run_event.clear()
        process_thread.join()
