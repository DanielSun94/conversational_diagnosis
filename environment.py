from config import logger


class Environment(object):
    def __init__(self, patient, doctor):
        self.patient_simulator = patient
        self.doctor_simulator = doctor

    def run(self, key, data):
        unified_id, data = self.patient_simulator.reset(key, data)
        self.doctor_simulator.reset()
        patient = self.patient_simulator
        doctor = self.doctor_simulator

        terminate = False
        data_list = []
        round_number = 0
        while not terminate:
            question, terminate = doctor.step(data_list)
            if not terminate:
                response = patient.response(question)
            else:
                response = ''
            logger.info('round: {}'.format(round_number))
            logger.info('key: {}'.format(key))
            logger.info('question: {}'.format(question))
            logger.info('response: {}'.format(response))
            round_number += 1
            data_list.append({
                'question': question,
                'response': response,
                'terminate': terminate,
            })
        if len(data_list) < 3:
            logger.info('key: {}, too short'.format(key))
        return {'unified_id': unified_id, 'emr': data, 'dialogue': data_list}


