from omegaconf import DictConfig
from argparse import Namespace


def get_model_name(args: DictConfig | Namespace) -> str:
    if isinstance(args, Namespace):
        if args.pretrained and len(args.output_prefix) == 0:
            return args.pretrained[:-3]
        else:
            if args.exposure == 'uniform':
                prefix = ''
            else:
                prefix = args.exposure + '_'
            prefix += '_'.join(sorted(args.source_id.split(',')))
            save_name = args.output_prefix + f'{prefix}_N{args.Neecr}_B{args.mf}'
            if args.threshold > 0:
                save_name += '_th' + str(args.threshold)
            save_name += '_sig{:.0f}'.format(100*args.sigmaLnE)
            if args.exclude_energy:
                save_name += '_noE'
            elif args.use_energy_as_feature:
                save_name += '_noEcoord'
            return save_name
    else:
        if args.model.pretrained and len(args.model.output_prefix) == 0:
            return args.model.pretrained[:-3]
        else:
            if args.data.exposure == 'uniform':
                prefix = ''
            else:
                prefix = args.data.exposure + '_'
            prefix += '_'.join(sorted(args.data.source_id.split(',')))
            save_name = args.model.output_prefix + f'{prefix}_N{args.data.Neecr}_B{args.data.mf}'
            if args.data.threshold > 0:
                save_name += '_th' + str(args.data.threshold)
            save_name += '_sig{:.0f}'.format(100*args.data.sigmaLnE)
            if args.data.exclude_energy:
                save_name += '_noE'
            elif args.model.use_energy_as_feature:
                save_name += '_noEcoord'
            return save_name